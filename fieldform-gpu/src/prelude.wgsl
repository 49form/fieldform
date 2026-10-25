// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// Runtime support for generated field shaders. Prepended to every program by `codegen`.
//
// Everything the generated `field()` function calls lives here. Keeping it hand-written rather
// than generated means the tricky parts -- texture addressing, the conservative answer outside
// a bake, the smooth minimum -- are written once and reviewed once, and the generated part
// stays a plain transcription of the graph.

// Numeric values for every parameter in the graph, literals included.
//
// This buffer is the reason the whole thing is worth building. A parameter change writes four
// bytes here and redraws; it does not rebuild geometry, retessellate a cage, or recompile this
// shader. Only a change to the graph's *shape* needs a new program.
@group(0) @binding(0) var<storage, read> params: array<f32>;

// Polyline spines for pipe nodes, packed end to end. `w` is the arc length from the curve's
// start to that point, so a walk that visits segments out of order -- the tree below -- can
// still say how far along the curve a hit is. After each spine's points come the nodes of its
// segment tree, two entries a node; see `nearest_segment`.
@group(0) @binding(1) var<storage, read> curve_points: array<vec4<f32>>;

// **Whether a profile or a wall is read from its outline rather than its bake** (backlog item 115).
//
// A bake is read a margin down so it never overstates, which puts its zero a margin outside the
// true outline all the way round: every baked body is drawn that much fat, and a knife edge shows
// it as a chamfer `margin / tan(half the edge's angle)` wide, however fine the bake. So the march
// uses the bake to get *near* the surface and the outline to land on it: `trace_primary` sets this
// at its first confirmed candidate when the frame asks (`Camera.jitter.z`, the still tier), and
// from there every reading of that ray -- the steps left, the confirmation, the settle, the
// normal -- walks the outline. Private, so the occlusion and the shadow, dispatched apart, keep
// the bake. Off, nothing changes anywhere.
var<private> exact_reads: bool = false;

// Where one curve's points sit, and — for a profile — the grid its distance was baked onto.
//
// The bake rides along with the span rather than in a table of its own because it is indexed by
// exactly the same number: the shader already asks "which curve", and this is one more fact about
// that curve. `bake_wide` of zero means there is none and the boundary is walked instead.
struct CurveSpan {
    start: u32,
    count: u32,
    // The bake, in two levels: a coarse grid over the whole box carrying sign and a safe bound,
    // and fine blocks stored only where the surface passes. `bake_wide` of zero means there is
    // none and the boundary is walked instead. See `baked_profile.rs` for why it is a band.
    coarse_start: u32,
    page_start: u32,
    fine_start: u32,
    blocks_wide: u32,
    blocks_high: u32,
    bake_wide: u32,
    bake_high: u32,
    lo_x: f32,
    lo_y: f32,
    cell: f32,
    margin: f32,
    coarse_margin: f32,
    // The segment tree over this curve, as node pairs in `curve_points`: where the first node
    // sits and how many there are. Zero nodes means the segments are walked one by one.
    bvh_start: u32,
    bvh_count: u32,
    // The bake's segment lists: per block cell a (start, count) pair from `segs_start`, the lists
    // from `seglist_start`, both in `profile_bakes`, and whether the outline turns anticlockwise.
    segs_start: u32,
    seglist_start: u32,
    ccw: u32,
}
@group(0) @binding(2) var<storage, read> curve_spans: array<CurveSpan>;

// How far from a block, in margins, its segments are listed -- `fieldform_field::LIST_REACH`,
// which says why four. A point whose nearest listed segment is within this of it has no nearer
// unlisted one.
const LIST_REACH: f32 = 4.0;

// Signed distance sampled on a grid, for every profile that has one, packed end to end.
@group(0) @binding(7) var<storage, read> profile_bakes: array<f32>;

// Where each baked cage sits, in world space and in the atlas.
struct CageSlab {
    // World-space corner of the bake, and the reciprocal of its cell size.
    lo: vec4<f32>,
    inv_cell: vec4<f32>,
    // Samples per axis in the bake.
    dims: vec4<u32>,
    // First atlas layer belonging to this cage, and how far values were clamped.
    layer_offset: u32,
    band: f32,
    _pad: vec2<f32>,
}
@group(0) @binding(3) var<storage, read> cage_slabs: array<CageSlab>;

// All baked cages stacked along Z into one 3D texture.
//
// An atlas rather than one texture per cage, because a binding array needs a wgpu feature that
// is not available everywhere, and a cage bake is small enough that stacking costs nothing.
//
// Read with `textureLoad` and interpolated by hand below, rather than with a sampler. A
// 32-bit float texture is not filterable without an optional feature, and the alternatives
// were worse: 16-bit floats would throw away precision the bake was careful to keep, and
// requiring the feature would make the viewport fail to start on hardware that is otherwise
// perfectly capable. Eight loads and seven mixes is a cheap price for running everywhere.
@group(1) @binding(0) var cage_atlas: texture_3d<f32>;

// Smooth minimum: a fillet of radius k, matching the CPU form in fieldform-field exactly.
//
// The two must agree: the CPU path meshes for export while this one draws, and a seam that
// moves between them is a shape that changes when you export it.
//
// At n = 2 the profile is a quarter circle of radius exactly k — see `smooth_min` in
// fieldform-field for the measurements. The exponent is passed rather than assumed so that
// G2, which is this same expression with a larger n, needs no second function.
fn smooth_min(a: f32, b: f32, k: f32, n: f32) -> f32 {
    if k <= 0.0 {
        return min(a, b);
    }
    let x = max(k - a, 0.0);
    let y = max(k - b, 0.0);
    // pow(0, n) is 0 and pow of a positive is well defined, so no branch is needed here.
    return max(k, min(a, b)) - pow(pow(x, n) + pow(y, n), 1.0 / n);
}

// What a blend reports, matching `blend_bound` in fieldform-field, where the argument is: the
// signed distance, in the max-norm of the two operand values, from `(a, b)` to the blend's own
// zero set. Nought exactly on the blended surface; never more than a millimetre a millimetre
// whatever the operands' gradients are doing, so never more than the distance to that surface;
// and **nothing divided**, so a blend of a blend is not short by a factor a link. A radius clear
// of the join it is the nearer operand, to the bit.
//
// The closed form is the circular fillet's. Any other exponent gets the old division, which is
// sound and is what it had; n = 3 and n = 4 each have a closed form to put here when G2 picks
// one.
//
// A chamfer -- exponent one -- has its own, simpler: the zero set's middle piece is the segment
// `x + y = k`, which the diagonal meets where the gaps sum to `k`.
fn blend_bound(a: f32, b: f32, k: f32, n: f32) -> f32 {
    if k <= 0.0 {
        return min(a, b);
    }
    if n == 1.0 {
        let x = k - a;
        let y = k - b;
        var nearest = min(max(abs(a), max(y, 0.0)), max(abs(b), max(x, 0.0)));
        if abs(x - y) <= k {
            nearest = min(nearest, abs(k - (x + y)) * 0.5);
        }
        return select(nearest, -nearest, max(x, 0.0) + max(y, 0.0) > k);
    }
    if n != 2.0 {
        return smooth_min(a, b, k, n) / pow(2.0, 1.0 / n);
    }
    let x = k - a;
    let y = k - b;
    // The two straight pieces of the zero set, `a = 0` and `b = 0`, each ending where the arc
    // takes over.
    var nearest = min(max(abs(a), max(y, 0.0)), max(abs(b), max(x, 0.0)));
    // The arc, where the diagonal through `(x, y)` crosses it.
    let across = x - y;
    if abs(across) <= k {
        let along = sqrt(2.0 * k * k - across * across);
        nearest = min(nearest, abs(along - (x + y)) * 0.5);
    }
    let inside = length(vec2<f32>(max(x, 0.0), max(y, 0.0))) > k;
    return select(nearest, -nearest, inside);
}

// The gradient of `blend_bound`, **with its length**, for the reason `blend_gradient` carries
// its own: a blend can be an operand, and the blend above it weighs the gradient it is handed.
// Whichever piece of the zero set is nearest decides it: along the arc a mix of the two operand
// gradients that sums to one, along a straight piece the one operand's own.
fn blend_bound_normal(
    a: f32, b: f32, ga: vec3<f32>, gb: vec3<f32>, k: f32, n: f32
) -> vec3<f32> {
    if k <= 0.0 {
        return select(gb, ga, a <= b);
    }
    if n == 1.0 {
        // The chamfer's three pieces, as `blend_bound` reads them: a ray's own operand, or
        // the bevel, whose gradient is the two operands' halves.
        let x = k - a;
        let y = k - b;
        let inside = max(x, 0.0) + max(y, 0.0) > k;
        let by_a = max(abs(a), max(y, 0.0));
        let by_b = max(abs(b), max(x, 0.0));
        var nearest = by_a;
        var gradient = select(select(-gb, gb, inside), ga, abs(a) >= max(y, 0.0));
        if by_b < nearest {
            nearest = by_b;
            gradient = select(select(-ga, ga, inside), gb, abs(b) >= max(x, 0.0));
        }
        if abs(x - y) <= k && abs(k - (x + y)) * 0.5 < nearest {
            gradient = 0.5 * (ga + gb);
            seam_scale = seam_scale_of(ga, gb);
        }
        return gradient;
    }
    if n != 2.0 {
        // Within both operands' reach the divided form mixes the two gradients as the arc
        // does, and reads short by about as much.
        if k - a > 0.0 && k - b > 0.0 {
            seam_scale = seam_scale_of(ga, gb);
        }
        return blend_gradient(a, b, ga, gb, k, k, n) / pow(2.0, 1.0 / n);
    }
    let x = k - a;
    let y = k - b;
    let inside = length(vec2<f32>(max(x, 0.0), max(y, 0.0))) > k;
    // Each candidate's distance, and the gradient of the *signed* value it would give.
    let by_a = max(abs(a), max(y, 0.0));
    let by_b = max(abs(b), max(x, 0.0));
    var nearest = by_a;
    // `a` itself where `a` is what the distance is; otherwise how far short of the piece's end
    // the point stands, which grows as `b` falls outside and as it rises inside.
    var gradient = select(select(-gb, gb, inside), ga, abs(a) >= max(y, 0.0));
    if by_b < nearest {
        nearest = by_b;
        gradient = select(select(-ga, ga, inside), gb, abs(b) >= max(x, 0.0));
    }
    let across = x - y;
    if abs(across) <= k {
        let along = sqrt(2.0 * k * k - across * across);
        if abs(along - (x + y)) * 0.5 < nearest {
            // d/d(x, y) of (along - x - y) / 2, and x, y fall as a, b rise.
            let lean = across / max(along, 1e-12);
            gradient = 0.5 * ((1.0 + lean) * ga + (1.0 - lean) * gb);
            seam_scale = seam_scale_of(ga, gb);
        }
    }
    return gradient;
}

// What a rounded blend's value is divided by, matching `blend_divisor` in fieldform-field.
//
// Inside its zone the smooth minimum rises 2^(1/n) per millimetre where the operand gradients
// are parallel -- a face the operands share -- so undivided it claims more room than there is
// and the march steps through the raised face (blend-integrity-spec, section 2.2). A positive
// constant moves no zero crossing, so the surface, the fillet and the normal are untouched; only
// the step length changes. Decided from the radius at run time rather than written as a literal,
// because the radius is a slot and dragging it to nought and back must not rebuild the shader.
fn blend_divisor(k: f32, n: f32) -> f32 {
    return select(1.0, pow(2.0, 1.0 / n), k > 0.0);
}

// The same for a rounded pattern, matching `pattern_join_divisor`: W^(1/n), W being the
// instances the window sums. All three of its ingredients are already slots -- the reach, the
// count and the first instance held -- so this costs no parameter and no recompile.
fn pattern_join_divisor(k: f32, reach: f32, count: f32, first: f32, n: f32) -> f32 {
    if k <= 0.0 {
        return 1.0;
    }
    let summed = max(min(2.0 * reach + 1.0, count - first), 1.0);
    return pow(summed, 1.0 / n);
}

// ---- Critical surfaces (blend-integrity-spec, section 6) ----
//
// How far the point `field` was last called with is from the nearest critical surface, less that
// surface's clearance. Set at the top of `field` and `field_normal` from the WORLD point and read
// by every tapering blend below them, however many placements, mirrors or pattern steps deep it
// sits: the exactness a critical surface is promised is pointwise in world space, so the gap must
// never be recomputed from whatever local point a nested blend happens to hold. A module variable
// rather than an argument for the reason `owner` is one -- threading it through every body and
// subtree function would rebuild the live range those functions exist to break. Its resting value
// is "nothing is near", which is every fillet at its full radius.
var<private> crit_gap: f32 = 3.4e38;
// Set by a blend's normal when the radius it used was on its way down. See `field_normal`.
var<private> taper_hit: bool = false;
// **What a reading along this hit's normal falls short by, for the still's occlusion to put
// back** (`round-chamfer-spec.md` §4.4). A blend's value is the max-norm distance to its zero set
// in the operands' plane, which along the seam's bisector moves by `cos(θ/2)` of a step, `θ` the
// seam's angle: 0.707 square, a half at 120°. The field is right to read so -- no two values can
// know the angle between their gradients -- but occlusion takes a short reading for nearby
// material and paints every fillet as a trench. The blend's normal knows both gradients, so it
// leaves `2 / |ĝa + ĝb|` here when the arc is the nearest piece, one otherwise; `field_normal`
// keeps the hit body's, `march` writes it beside the pixel, and `occlude` multiplies its five
// readings by it. An estimate, as occlusion is, and exact on the bisector; the shadow is left
// alone, since its march goes toward the light and not along the normal.
var<private> seam_scale: f32 = 1.0;

// `2 / |ĝa + ĝb|`, which is `1 / cos(θ/2)`, capped: past 120° a seam is nearly flat and a
// reading scaled three times is a crevice erased, which is the wrong picture the other way.
fn seam_scale_of(ga: vec3<f32>, gb: vec3<f32>) -> f32 {
    let la = length(ga);
    let lb = length(gb);
    if la < 1e-9 || lb < 1e-9 {
        return 1.0;
    }
    return clamp(2.0 / max(length(ga / la + gb / lb), 1e-6), 1.0, 3.0);
}

// The radius a fillet gets here: nought on a critical surface and across its clearance band, the
// whole radius once the gap exceeds it. Matches `tapered` in fieldform-field.
fn tapered(k: f32) -> f32 {
    return min(max(crit_gap, 0.0), k);
}

// `smooth_min` for a radius that may be on its way to nought, matching `smooth_min_tapering`.
//
// NOT `min(a, b)` at a tapered radius of nought, which is what `smooth_min` answers for a radius
// typed as nought: that is not the limit of the form where both operands are negative, and a
// radius that *reaches* nought at the edge of a clearance band would step there. The form is kept
// all the way down. Its zero set at nought is exactly the hard boolean's.
fn smooth_min_tapering(a: f32, b: f32, k: f32, n: f32) -> f32 {
    if k <= 0.0 {
        return min(a, b);
    }
    let ke = tapered(k);
    let x = max(ke - a, 0.0);
    let y = max(ke - b, 0.0);
    return max(ke, min(a, b)) - pow(pow(x, n) + pow(y, n), 1.0 / n);
}

// The tapering blend's divisor: `rho` is how fast the gap can move over how fast the operands
// can, asked of `fieldform_field::taper_ratio` so both paths divide by one number. At rho = 1 and
// n = 2 this is 1.83; the proof is beside `taper_ratio`.
fn blend_taper_divisor(k: f32, n: f32, rho: f32) -> f32 {
    let steepest = pow(2.0, 1.0 / n);
    return select(1.0, steepest + (steepest - 1.0) * rho, k > 0.0);
}

fn pattern_taper_divisor(k: f32, reach: f32, count: f32, first: f32, n: f32, rho: f32) -> f32 {
    if k <= 0.0 {
        return 1.0;
    }
    let steepest = pow(max(min(2.0 * reach + 1.0, count - first), 1.0), 1.0 / n);
    return steepest + (steepest - 1.0) * rho;
}

// The tapering blend's normal, and a note that it was tapering.
//
// A radius that varies with position adds a term to the gradient -- (1 - s) times the gradient of
// the gap -- and that gradient lives in world space while this blend may be three placements and
// a mirror deep. Rather than carry a world vector down through every kind of frame, the blend
// says it was tapering and `field_normal` differences the field for that pixel, which is right
// under any nesting and is exactly what the CPU does for every blend. Only pixels on a fillet
// that is actually thinning pay for it.
fn smooth_min_normal_tapering(
    a: f32, b: f32, ga: vec3<f32>, gb: vec3<f32>, k: f32, n: f32
) -> vec3<f32> {
    let ke = tapered(k);
    if k > 0.0 && ke < k && a < ke && b < ke {
        taper_hit = true;
    }
    return blend_gradient(a, b, ga, gb, k, ke, n);
}

// The gradient of a blend's *raw* form, with its length: the operand gradients under weights of
// unit conjugate norm. `k` decides whether this is a blend at all and `ke` is the radius used.
//
// **With its length, because a blend can be an operand.** The weights assume `ga` and `gb` are
// the operands' real gradients. A blend's value is divided so that it never overstates, so its
// gradient is short by that divisor -- and handing an outer blend a *unit* vector for it
// over-weights it by the same factor. Measured on a bore through a body of nested fillets, the
// normalized form read 8.6 degrees off at the 99th percentile and 12 at worst; the caller divides
// this by the blend's divisor and `field_normal` normalizes once, at the end.
fn blend_gradient(
    a: f32, b: f32, ga: vec3<f32>, gb: vec3<f32>, k: f32, ke: f32, n: f32
) -> vec3<f32> {
    if k <= 0.0 {
        return select(gb, ga, a <= b);
    }
    let x = max(ke - a, 0.0);
    let y = max(ke - b, 0.0);
    if x <= 0.0 && y <= 0.0 {
        return select(gb, ga, a <= b);
    }
    let norm = pow(pow(x, n) + pow(y, n), 1.0 / n);
    // **A gap of nought weighs nothing, said rather than computed**: at a chamfer's exponent of
    // one the weight is `pow(0, 0)`, which WGSL leaves undefined and a driver may answer NaN.
    // Above one the power is nought there anyway, so this changes nothing a fillet drew.
    let wx = select(0.0, pow(x, n - 1.0), x > 0.0);
    let wy = select(0.0, pow(y, n - 1.0), y > 0.0);
    return (wx * ga + wy * gb) / pow(norm, n - 1.0);
}

// A field read as a number and mapped onto a range, clamped at both ends.
//
// `shape` is 0 linear, 1 smoothstep, 2 smootherstep — an integer in a float because the shader
// takes no enums, and it is a constant this generator writes rather than a parameter.
//
// The clamping is not cosmetic. It is what bounds the slope, and the bound is what the value
// above a driven parameter gets divided by; a mapping that kept going outside its input range
// would have no bound to state. `in_span` is never zero — an empty range is refused at compile
// time on both paths, for the same reason.
fn ramp(value: f32, in_min: f32, in_span: f32, out_min: f32, out_span: f32, shape: f32) -> f32 {
    let t = clamp((value - in_min) / in_span, 0.0, 1.0);
    var eased = t;
    if (shape > 1.5) {
        eased = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    } else if (shape > 0.5) {
        eased = t * t * (3.0 - 2.0 * t);
    }
    return out_min + out_span * eased;
}

// The blend's own normal, which is not the mix the polynomial form used.
//
// In the fillet, f = k - ||(x, y)||_n with x = (k-a)+ and y = (k-b)+, so df/dp is the two
// operand gradients weighted by how much each is "inside" the blend. At n = 2 that is exactly
// (x*ga + y*gb) / ||(x, y)||; the general form weights by the n-1 powers. Outside the fillet
// one of the weights is zero and this reduces to the active operand's own normal, which is
// what a hard boolean would have given.
fn smooth_min_normal(
    a: f32, b: f32, ga: vec3<f32>, gb: vec3<f32>, k: f32, n: f32
) -> vec3<f32> {
    return blend_gradient(a, b, ga, gb, k, k, n);
}

// Placement, matching `euler_degrees_to_rotation` and `to_local` in graph_field.rs.
//
// The angles arrive as parameters rather than as a baked matrix so that turning a body costs
// a buffer write instead of a shader recompile, which is the same trade every other number in
// the generated code makes. The cost is the six trig calls below, paid once per transform per
// sample rather than once per leaf.
fn euler_rotation(degrees: vec3<f32>) -> mat3x3<f32> {
    let r = radians(degrees);
    let c = cos(r);
    let s = sin(r);
    // Rz * Ry * Rx applied to a column vector -- nalgebra's roll/pitch/yaw order, taken rather
    // than invented so this matrix and the CPU's are the same one. WGSL builds a matrix from
    // its columns, so each row below is a column of that product.
    return mat3x3<f32>(
        vec3<f32>(c.y * c.z, c.y * s.z, -s.y),
        vec3<f32>(
            s.x * s.y * c.z - c.x * s.z,
            s.x * s.y * s.z + c.x * c.z,
            s.x * c.y,
        ),
        vec3<f32>(
            c.x * s.y * c.z + s.x * s.z,
            c.x * s.y * s.z - s.x * c.z,
            c.x * c.y,
        ),
    );
}

// Maps a world point into the space a placed body is defined in: undo the translation, then
// the rotation, then the scale.
fn to_local(
    p: vec3<f32>,
    translation: vec3<f32>,
    euler_degrees: vec3<f32>,
    scale: f32,
) -> vec3<f32> {
    // A rotation's transpose is its inverse, and `v * m` in WGSL is `transpose(m) * v`, so
    // multiplying from the left of the matrix is the inverse rotation without building one.
    return ((p - translation) * euler_rotation(euler_degrees)) / scale;
}

// The same, with the turn and the scale taken about `pivot` rather than about the local origin.
//
// A pivot is a fixed point: forward, this is `pivot + scale*m*(x - pivot) + translation`, and
// what follows is that undone. With the pivot at the origin it is exactly `to_local`, which is
// what every placement that does not use one still gets -- the two extra vector adds cost
// nothing beside the matrix, and building the matrix once here is why this is a function rather
// than an expression at the call site.
fn to_local_about(
    p: vec3<f32>,
    translation: vec3<f32>,
    euler_degrees: vec3<f32>,
    scale: f32,
    pivot: vec3<f32>,
) -> vec3<f32> {
    return ((p - translation - pivot) * euler_rotation(euler_degrees)) / scale + pivot;
}

// The way back out of that space, for anything that has to do work in a frame and hand the
// result to something reading world coordinates.
fn from_local(
    p: vec3<f32>,
    translation: vec3<f32>,
    euler_degrees: vec3<f32>,
    scale: f32,
) -> vec3<f32> {
    return euler_rotation(euler_degrees) * (p * scale) + translation;
}

// Closed-form solids, matching `primitive.rs` in fieldform-field exactly. Both are exact
// distances rather than bounds, which is what lets a blend fillet them correctly.
fn sphere_distance(p: vec3<f32>, centre: vec3<f32>, radius: f32) -> f32 {
    return length(p - centre) - radius;
}

fn sphere_normal(p: vec3<f32>, centre: vec3<f32>) -> vec3<f32> {
    let offset = p - centre;
    let length_of = length(offset);
    // At the exact centre every direction is equally correct; +X is as good as any and never
    // appears on a surface.
    if length_of < 1e-9 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return offset / length_of;
}

fn cuboid_distance(p: vec3<f32>, centre: vec3<f32>, size: vec3<f32>) -> f32 {
    let d = abs(p - centre) - size * 0.5;
    // Outside: the length of the positive part, so corners measure diagonally. Inside: the
    // least negative axis, which is the distance to the nearest face.
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn cuboid_normal(p: vec3<f32>, centre: vec3<f32>, size: vec3<f32>) -> vec3<f32> {
    let offset = p - centre;
    let d = abs(offset) - size * 0.5;
    let sign_of = sign(offset + vec3<f32>(1e-9));
    if max(d.x, max(d.y, d.z)) > 0.0 {
        return normalize(sign_of * max(d, vec3<f32>(0.0)));
    }
    // Inside: whichever face is nearest.
    if d.x >= d.y && d.x >= d.z {
        return vec3<f32>(sign_of.x, 0.0, 0.0);
    }
    if d.y >= d.z {
        return vec3<f32>(0.0, sign_of.y, 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign_of.z);
}

// **A box's or a cylinder's edges cut back**, matching `edged` in `primitive.rs` term for term:
// `edge` is the reach and the shape's exponent. Rounded is the inner solid's reading with the
// n-norm, `|q+|_n + min(max q, 0) - k` with `q = d + k` -- the rounded box itself at n = 2, the G2
// blend's curve across each edge at n = 3 -- and a chamfer is the solid intersected with a bevel
// across each edge, `(d_i + d_j + k) / sqrt(2)`. A hard edge never calls these: the plain
// functions above are what it draws, so a document without one compiles as it did.
fn edged3(d: vec3<f32>, edge: vec2<f32>) -> f32 {
    let exact = length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
    let k = edge.x;
    if k <= 0.0 {
        return exact;
    }
    if edge.y == 1.0 {
        let s = 0.70710678;
        return max(
            exact,
            max((d.x + d.y + k) * s, max((d.y + d.z + k) * s, (d.x + d.z + k) * s))
        );
    }
    let q = d + vec3<f32>(k);
    let o = max(q, vec3<f32>(0.0));
    let n = edge.y;
    let norm = pow(pow(o.x, n) + pow(o.y, n) + pow(o.z, n), 1.0 / n);
    return norm + min(max(q.x, max(q.y, q.z)), 0.0) - k;
}

fn edged2(d: vec2<f32>, edge: vec2<f32>) -> f32 {
    let exact = length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
    let k = edge.x;
    if k <= 0.0 {
        return exact;
    }
    if edge.y == 1.0 {
        return max(exact, (d.x + d.y + k) * 0.70710678);
    }
    let q = d + vec2<f32>(k);
    let o = max(q, vec2<f32>(0.0));
    let n = edge.y;
    let norm = pow(pow(o.x, n) + pow(o.y, n), 1.0 / n);
    return norm + min(max(q.x, q.y), 0.0) - k;
}

fn cuboid_edge_distance(p: vec3<f32>, centre: vec3<f32>, size: vec3<f32>, edge: vec2<f32>) -> f32 {
    return edged3(abs(p - centre) - size * 0.5, edge);
}

// Differenced, as every bound here is: the rounding's own normal, without a second closed form
// to keep agreeing with the first.
fn cuboid_edge_normal(p: vec3<f32>, centre: vec3<f32>, size: vec3<f32>, edge: vec2<f32>) -> vec3<f32> {
    let h = 1e-3;
    let n = vec3<f32>(
        cuboid_edge_distance(p + vec3<f32>(h, 0.0, 0.0), centre, size, edge)
            - cuboid_edge_distance(p - vec3<f32>(h, 0.0, 0.0), centre, size, edge),
        cuboid_edge_distance(p + vec3<f32>(0.0, h, 0.0), centre, size, edge)
            - cuboid_edge_distance(p - vec3<f32>(0.0, h, 0.0), centre, size, edge),
        cuboid_edge_distance(p + vec3<f32>(0.0, 0.0, h), centre, size, edge)
            - cuboid_edge_distance(p - vec3<f32>(0.0, 0.0, h), centre, size, edge),
    );
    let l = length(n);
    if l < 1e-12 {
        return cuboid_normal(p, centre, size);
    }
    return n / l;
}

// A box whose faces are not square to one another, by its three full edge vectors. The same
// arithmetic as `SkewedBox` in `primitive.rs`: with `t = inverse * (p - centre)` the box is
// `|t_i| <= 1/2`, and row i of the inverse -- the gradient of t_i, `cross(e_j, e_k) / det` -- makes
// the distance to a face plane `(t_i - c) / |g_i|`. Inside that is the whole answer; outside the
// nearest surface point is the foot of a perpendicular that lands inside its face, or the nearest
// point of one of twelve edges (corners included), and the least of those is the distance.
//
// Written without loops or arrays: every candidate is a call, so nothing here depends on the
// compiler unrolling anything, which the HLSL compiler on some machines will not do.

// Offers `at` as the nearest surface point if it beats `best`, which carries its squared
// distance in `.x` and the point in `.yzw` (relative to the centre).
fn skewed_offer(q: vec3<f32>, at: vec3<f32>, best: vec4<f32>) -> vec4<f32> {
    let d = dot(q - at, q - at);
    if d < best.x {
        return vec4<f32>(d, at);
    }
    return best;
}

// The foot of the perpendicular to the face `t_i = c`, taken only where it falls inside the
// parallelogram -- that is, where the other two coordinates are within half an edge. The foot's
// other coordinates follow from `t_j - dot(g_j, g_i) * s`, so no second inverse is needed.
fn skewed_face(
    q: vec3<f32>,
    best: vec4<f32>,
    gi: vec3<f32>,
    gj: vec3<f32>,
    gk: vec3<f32>,
    ti: f32,
    tj: f32,
    tk: f32,
    c: f32,
) -> vec4<f32> {
    let s = (ti - c) / dot(gi, gi);
    let fj = tj - dot(gj, gi) * s;
    let fk = tk - dot(gk, gi) * s;
    if abs(fj) <= 0.5 + 1e-5 && abs(fk) <= 0.5 + 1e-5 {
        return skewed_offer(q, q - gi * s, best);
    }
    return best;
}

// The nearest point of one edge, a segment from `start` along `along`; clamping to its ends is
// what makes the corners candidates too.
fn skewed_edge(q: vec3<f32>, best: vec4<f32>, start: vec3<f32>, along: vec3<f32>) -> vec4<f32> {
    let u = clamp(dot(q - start, along) / dot(along, along), 0.0, 1.0);
    return skewed_offer(q, start + along * u, best);
}

// The nearest point of the surface to a point outside, relative to the centre.
fn skewed_box_nearest(q: vec3<f32>, e0: vec3<f32>, e1: vec3<f32>, e2: vec3<f32>) -> vec3<f32> {
    let det = dot(e0, cross(e1, e2));
    let g0 = cross(e1, e2) / det;
    let g1 = cross(e2, e0) / det;
    let g2 = cross(e0, e1) / det;
    let t0 = dot(g0, q);
    let t1 = dot(g1, q);
    let t2 = dot(g2, q);
    var best = vec4<f32>(3.4e38, 0.0, 0.0, 0.0);
    best = skewed_face(q, best, g0, g1, g2, t0, t1, t2, -0.5);
    best = skewed_face(q, best, g0, g1, g2, t0, t1, t2, 0.5);
    best = skewed_face(q, best, g1, g2, g0, t1, t2, t0, -0.5);
    best = skewed_face(q, best, g1, g2, g0, t1, t2, t0, 0.5);
    best = skewed_face(q, best, g2, g0, g1, t2, t0, t1, -0.5);
    best = skewed_face(q, best, g2, g0, g1, t2, t0, t1, 0.5);
    // Edges along e0, then e1, then e2: the four positions of the two other coordinates each.
    let h0 = e0 * 0.5;
    let h1 = e1 * 0.5;
    let h2 = e2 * 0.5;
    best = skewed_edge(q, best, -h0 - h1 - h2, e0);
    best = skewed_edge(q, best, -h0 - h1 + h2, e0);
    best = skewed_edge(q, best, -h0 + h1 - h2, e0);
    best = skewed_edge(q, best, -h0 + h1 + h2, e0);
    best = skewed_edge(q, best, -h1 - h2 - h0, e1);
    best = skewed_edge(q, best, -h1 - h2 + h0, e1);
    best = skewed_edge(q, best, -h1 + h2 - h0, e1);
    best = skewed_edge(q, best, -h1 + h2 + h0, e1);
    best = skewed_edge(q, best, -h2 - h0 - h1, e2);
    best = skewed_edge(q, best, -h2 - h0 + h1, e2);
    best = skewed_edge(q, best, -h2 + h0 - h1, e2);
    best = skewed_edge(q, best, -h2 + h0 + h1, e2);
    return best.yzw;
}

fn skewed_box_distance(
    p: vec3<f32>,
    centre: vec3<f32>,
    e0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
) -> f32 {
    let q = p - centre;
    let det = dot(e0, cross(e1, e2));
    let n0 = cross(e1, e2);
    let n1 = cross(e2, e0);
    let n2 = cross(e0, e1);
    // |g_i| = |n_i| / |det|, so the slab distance is (|t_i| - 1/2) * |det| / |n_i|, and t_i * |n_i|
    // is dot(n_i, q) / det.
    let a = abs(det);
    let s0 = (abs(dot(n0, q) / det) - 0.5) * a / length(n0);
    let s1 = (abs(dot(n1, q) / det) - 0.5) * a / length(n1);
    let s2 = (abs(dot(n2, q) / det) - 0.5) * a / length(n2);
    let nearest_face = max(s0, max(s1, s2));
    if nearest_face <= 0.0 {
        return nearest_face;
    }
    return length(q - skewed_box_nearest(q, e0, e1, e2));
}

fn skewed_box_normal(
    p: vec3<f32>,
    centre: vec3<f32>,
    e0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
) -> vec3<f32> {
    let q = p - centre;
    let det = dot(e0, cross(e1, e2));
    let n0 = cross(e1, e2);
    let n1 = cross(e2, e0);
    let n2 = cross(e0, e1);
    let a = abs(det);
    let t0 = dot(n0, q) / det;
    let t1 = dot(n1, q) / det;
    let t2 = dot(n2, q) / det;
    let s0 = (abs(t0) - 0.5) * a / length(n0);
    let s1 = (abs(t1) - 0.5) * a / length(n1);
    let s2 = (abs(t2) - 0.5) * a / length(n2);
    let nearest_face = max(s0, max(s1, s2));
    if nearest_face <= 0.0 {
        // Inside: out through the nearest face, along that face's gradient.
        // `n_i / det` is g_i; its sign is the sign of t_i's side, and only the direction matters.
        let flip = select(-1.0, 1.0, det > 0.0);
        if s0 >= s1 && s0 >= s2 {
            return normalize(n0 * flip * select(-1.0, 1.0, t0 >= 0.0));
        }
        if s1 >= s2 {
            return normalize(n1 * flip * select(-1.0, 1.0, t1 >= 0.0));
        }
        return normalize(n2 * flip * select(-1.0, 1.0, t2 >= 0.0));
    }
    let out = q - skewed_box_nearest(q, e0, e1, e2);
    return out / max(length(out), 1e-9);
}

// The round solids. Three of them are drawn between two points and one about an axis, which
// is the same split `primitive.rs` makes and for the same reason: a cylinder, cone or capsule
// has two ends to name, and a torus does not.
//
// Each two-point shape is evaluated in the frame where its segment becomes +Z, so the actual
// distance is written once — in the axis-aligned form below — rather than twice. The frame's
// two perpendiculars are picked arbitrarily, which is sound because every shape using it is
// rotationally symmetric about the segment: a different choice rotates the local point and
// rotates the normal back by the same amount. That is also why this may pick its perpendicular
// differently from the CPU and still agree in world space.

// The direction straight out from the +Z axis. On the axis there is no such direction; +X is
// as good as any and the callers below only use it where the radial part is non-zero.
fn radial_direction(q: vec3<f32>) -> vec3<f32> {
    let rho = length(q.xy);
    if rho < 1e-9 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return vec3<f32>(q.x / rho, q.y / rho, 0.0);
}

// Columns are the frame's u, v, w: across, across, and along the segment.
fn segment_frame(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    let along = b - a;
    let len = length(along);
    if len < 1e-9 {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let w = along / len;
    // Cross with whichever axis `w` leans on least, so the product is never near zero.
    var least = vec3<f32>(0.0, 0.0, 1.0);
    if abs(w.x) <= abs(w.y) && abs(w.x) <= abs(w.z) {
        least = vec3<f32>(1.0, 0.0, 0.0);
    } else if abs(w.y) <= abs(w.z) {
        least = vec3<f32>(0.0, 1.0, 0.0);
    }
    let u = normalize(cross(w, least));
    return mat3x3<f32>(u, cross(w, u), w);
}

fn segment_local(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let frame = segment_frame(a, b);
    let q = p - (a + b) * 0.5;
    return vec3<f32>(dot(q, frame[0]), dot(q, frame[1]), dot(q, frame[2]));
}

fn segment_world(n: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let frame = segment_frame(a, b);
    return frame[0] * n.x + frame[1] * n.y + frame[2] * n.z;
}

// --- axis-aligned cores, matching primitive.rs term for term ---

fn axis_cylinder_distance(p: vec3<f32>, radius: f32, height: f32) -> f32 {
    let d = vec2<f32>(length(p.xy) - radius, abs(p.z) - height * 0.5);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn axis_cylinder_normal(p: vec3<f32>, radius: f32, height: f32) -> vec3<f32> {
    let d = vec2<f32>(length(p.xy) - radius, abs(p.z) - height * 0.5);
    let axis = vec3<f32>(0.0, 0.0, select(1.0, -1.0, p.z < 0.0));
    if d.x > 0.0 || d.y > 0.0 {
        let o = max(d, vec2<f32>(0.0));
        let len = max(length(o), 1e-9);
        return radial_direction(p) * (o.x / len) + axis * (o.y / len);
    }
    if d.x >= d.y {
        return radial_direction(p);
    }
    return axis;
}

// The offsets to the nearer cap and to the slanted side, in the (radial, axial) plane. The
// shorter decides both the distance and the normal, so they are found together.
fn axis_cone_features(p: vec3<f32>, r_bottom: f32, r_top: f32, height: f32) -> mat2x2<f32> {
    let half = height * 0.5;
    let rho = length(p.xy);
    let cap_radius = select(r_top, r_bottom, p.z < 0.0);
    let ca = vec2<f32>(rho - min(rho, cap_radius), abs(p.z) - half);

    let k1 = vec2<f32>(r_top, half);
    let k2 = vec2<f32>(r_top - r_bottom, 2.0 * half);
    let along = dot(k2, k2);
    var t = 0.0;
    if along > 0.0 {
        t = clamp(dot(k1 - vec2<f32>(rho, p.z), k2) / along, 0.0, 1.0);
    }
    let cb = vec2<f32>(rho, p.z) - k1 + k2 * t;
    return mat2x2<f32>(ca, cb);
}

fn axis_cone_sign(ca: vec2<f32>, cb: vec2<f32>) -> f32 {
    return select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
}

fn axis_cone_distance(p: vec3<f32>, r_bottom: f32, r_top: f32, height: f32) -> f32 {
    let f = axis_cone_features(p, r_bottom, r_top, height);
    return axis_cone_sign(f[0], f[1]) * sqrt(min(dot(f[0], f[0]), dot(f[1], f[1])));
}

fn axis_cone_normal(p: vec3<f32>, r_bottom: f32, r_top: f32, height: f32) -> vec3<f32> {
    let f = axis_cone_features(p, r_bottom, r_top, height);
    let da = dot(f[0], f[0]);
    let db = dot(f[1], f[1]);
    let cap_is_nearer = da < db;
    let offset = select(f[1], f[0], cap_is_nearer);
    let len = max(sqrt(min(da, db)), 1e-9);
    // The cap offset was measured with the two caps folded together, so its axial part is
    // unfolded back onto the side the point is actually on.
    let axial = select(offset.y, -offset.y, cap_is_nearer && p.z < 0.0);
    return (radial_direction(p) * offset.x + vec3<f32>(0.0, 0.0, axial))
        * (axis_cone_sign(f[0], f[1]) / len);
}

// A capsule about Z with a radius at each end: two spheres and the cone tangent to both.
// `x` and `y` are the tangent's direction in the (radial, axial) plane; when one sphere
// swallows the other there is no tangent and the shape is the union of the two spheres.
fn axis_capsule_swallowed(r_start: f32, r_end: f32, height: f32) -> bool {
    return height < 1e-9 || abs(r_start - r_end) >= height;
}

fn axis_capsule_distance(p: vec3<f32>, r_start: f32, r_end: f32, height: f32) -> f32 {
    let h = height * 0.5;
    let low = length(p - vec3<f32>(0.0, 0.0, -h)) - r_start;
    let high = length(p - vec3<f32>(0.0, 0.0, h)) - r_end;
    if axis_capsule_swallowed(r_start, r_end, height) {
        return min(low, high);
    }
    let q = vec2<f32>(length(p.xy), p.z + h);
    let b = (r_start - r_end) / height;
    let a = sqrt(max(1.0 - b * b, 0.0));
    let k = dot(q, vec2<f32>(-b, a));
    if k < 0.0 {
        return length(q) - r_start;
    }
    if k > a * height {
        return length(q - vec2<f32>(0.0, height)) - r_end;
    }
    return dot(q, vec2<f32>(a, b)) - r_start;
}

fn axis_capsule_normal(p: vec3<f32>, r_start: f32, r_end: f32, height: f32) -> vec3<f32> {
    let h = height * 0.5;
    if axis_capsule_swallowed(r_start, r_end, height) {
        let low = length(p - vec3<f32>(0.0, 0.0, -h)) - r_start;
        let high = length(p - vec3<f32>(0.0, 0.0, h)) - r_end;
        let centre = vec3<f32>(0.0, 0.0, select(h, -h, low <= high));
        let offset = p - centre;
        return offset / max(length(offset), 1e-9);
    }
    let q = vec2<f32>(length(p.xy), p.z + h);
    let b = (r_start - r_end) / height;
    let a = sqrt(max(1.0 - b * b, 0.0));
    let k = dot(q, vec2<f32>(-b, a));
    var d = vec2<f32>(a, b);
    if k < 0.0 {
        d = q / max(length(q), 1e-9);
    } else if k > a * height {
        let o = q - vec2<f32>(0.0, height);
        d = o / max(length(o), 1e-9);
    }
    return radial_direction(p) * d.x + vec3<f32>(0.0, 0.0, d.y);
}

// --- the two-point forms the graph actually calls ---

fn cylinder_distance(p: vec3<f32>, base: vec3<f32>, top: vec3<f32>, radius: f32) -> f32 {
    return axis_cylinder_distance(segment_local(p, base, top), radius, distance(base, top));
}

fn cylinder_normal(p: vec3<f32>, base: vec3<f32>, top: vec3<f32>, radius: f32) -> vec3<f32> {
    let n = axis_cylinder_normal(segment_local(p, base, top), radius, distance(base, top));
    return segment_world(n, base, top);
}

// Both circular edges cut back, read in the meridian half-plane as `EdgedCylinder` reads them.
fn cylinder_edge_distance(
    p: vec3<f32>,
    base: vec3<f32>,
    top: vec3<f32>,
    radius: f32,
    edge: vec2<f32>,
) -> f32 {
    let q = segment_local(p, base, top);
    let d = vec2<f32>(length(q.xy) - radius, abs(q.z) - distance(base, top) * 0.5);
    return edged2(d, edge);
}

fn cylinder_edge_normal(
    p: vec3<f32>,
    base: vec3<f32>,
    top: vec3<f32>,
    radius: f32,
    edge: vec2<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    let n = vec3<f32>(
        cylinder_edge_distance(p + vec3<f32>(h, 0.0, 0.0), base, top, radius, edge)
            - cylinder_edge_distance(p - vec3<f32>(h, 0.0, 0.0), base, top, radius, edge),
        cylinder_edge_distance(p + vec3<f32>(0.0, h, 0.0), base, top, radius, edge)
            - cylinder_edge_distance(p - vec3<f32>(0.0, h, 0.0), base, top, radius, edge),
        cylinder_edge_distance(p + vec3<f32>(0.0, 0.0, h), base, top, radius, edge)
            - cylinder_edge_distance(p - vec3<f32>(0.0, 0.0, h), base, top, radius, edge),
    );
    let l = length(n);
    if l < 1e-12 {
        return cylinder_normal(p, base, top, radius);
    }
    return n / l;
}

// **A cone's edges cut back** -- `EdgedCone` in `primitive.rs`, term for term. Its meridian
// section is a convex polygon, kept as the chain of its right-hand vertices from the axis at the
// bottom round to the axis at the top, and that polygon's distance at `(rho, z)` is the solid's
// in space: a round is the inset polygon's distance less the reach, a chamfer the clipped
// polygon's, both exact; G2 is the hard polygon held against the blend family's join at each
// corner, a bound. Up to six vertices, built per sample from the cone's numbers, since which
// edges survive a cut depends on the reach being dragged.
struct Meridian {
    v: array<vec2<f32>, 6>,
    n: u32,
}

fn meridian_distance(m: Meridian, q: vec2<f32>) -> f32 {
    var nearest = 1e30;
    var depth = 1e30;
    var inside = true;
    for (var i = 0u; i + 1u < m.n; i = i + 1u) {
        let a = m.v[i];
        let b = m.v[i + 1u];
        let d = b - a;
        let len = length(d);
        if (len < 1e-6) {
            nearest = min(nearest, length(q - a));
            continue;
        }
        let t = clamp(dot(q - a, d) / (len * len), 0.0, 1.0);
        nearest = min(nearest, length(q - (a + d * t)));
        // Outward: the polygon lies to the left of a chain that runs anticlockwise.
        let s = dot(q - a, vec2<f32>(d.y, -d.x)) / len;
        if (s > 0.0) {
            inside = false;
        }
        depth = min(depth, -s);
    }
    return select(nearest, -depth, inside);
}

// The slant's outward unit normal in `xy` and its offset in `z`: `n . q = c` on the slant.
fn cone_slant(half: f32, rb: f32, rt: f32) -> vec3<f32> {
    let n = vec2<f32>(2.0 * half, rb - rt) / length(vec2<f32>(rb - rt, 2.0 * half));
    return vec3<f32>(n, dot(n, vec2<f32>(rb, -half)));
}

fn cone_hard_meridian(half: f32, rb: f32, rt: f32) -> Meridian {
    var m: Meridian;
    m.v[0] = vec2<f32>(0.0, -half);
    m.v[1] = vec2<f32>(rb, -half);
    m.v[2] = vec2<f32>(rt, half);
    m.v[3] = vec2<f32>(0.0, half);
    m.n = 4u;
    return m;
}

// Every edge moved in by `k`: a trapezoid, or a triangle where the moved slants meet on the axis
// before the moved cap. The compile refused a reach that leaves nothing.
fn cone_inset_meridian(half: f32, rb: f32, rt: f32, k: f32) -> Meridian {
    let s = cone_slant(half, rb, rt);
    let zb = -half + k;
    let zt = half - k;
    let ra = (s.z - k - s.y * zb) / s.x;
    let rtt = (s.z - k - s.y * zt) / s.x;
    var m: Meridian;
    if (ra >= 0.0 && rtt >= 0.0) {
        m.v[0] = vec2<f32>(0.0, zb);
        m.v[1] = vec2<f32>(ra, zb);
        m.v[2] = vec2<f32>(rtt, zt);
        m.v[3] = vec2<f32>(0.0, zt);
        m.n = 4u;
    } else if (rtt < 0.0) {
        m.v[0] = vec2<f32>(0.0, zb);
        m.v[1] = vec2<f32>(ra, zb);
        m.v[2] = vec2<f32>(0.0, (s.z - k) / s.y);
        m.n = 3u;
    } else {
        m.v[0] = vec2<f32>(0.0, (s.z - k) / s.y);
        m.v[1] = vec2<f32>(rtt, zt);
        m.v[2] = vec2<f32>(0.0, zt);
        m.n = 3u;
    }
    return m;
}

// Each corner cut back `k` along both its edges: a bevel across a circular edge, a flat across an
// apex.
fn cone_clipped_meridian(half: f32, rb: f32, rt: f32, k: f32) -> Meridian {
    let a = vec2<f32>(rb, -half);
    let b = vec2<f32>(rt, half);
    let t = (b - a) / length(b - a);
    var m: Meridian;
    var n = 0u;
    if (rb > 0.0) {
        m.v[0] = vec2<f32>(0.0, -half);
        m.v[1] = vec2<f32>(rb - k, -half);
        n = 2u;
    } else {
        m.v[0] = vec2<f32>(0.0, (a + t * k).y);
        n = 1u;
    }
    m.v[n] = a + t * k;
    m.v[n + 1u] = b - t * k;
    n = n + 2u;
    if (rt > 0.0) {
        m.v[n] = vec2<f32>(rt - k, half);
        m.v[n + 1u] = vec2<f32>(0.0, half);
        n = n + 2u;
    } else {
        m.v[n] = vec2<f32>(0.0, (b - t * k).y);
        n = n + 1u;
    }
    m.n = n;
    return m;
}

fn cone_edge_distance(
    p: vec3<f32>,
    base: vec3<f32>,
    top: vec3<f32>,
    r_base: f32,
    r_top: f32,
    edge: vec2<f32>,
) -> f32 {
    let local = segment_local(p, base, top);
    let q = vec2<f32>(length(local.xy), local.z);
    let half = distance(base, top) * 0.5;
    if (edge.y == 1.0) {
        return meridian_distance(cone_clipped_meridian(half, r_base, r_top, edge.x), q);
    }
    if (edge.y == 2.0) {
        return meridian_distance(cone_inset_meridian(half, r_base, r_top, edge.x), q) - edge.x;
    }
    // The family's join at each corner: the slant with a cap, or with its own mirror image at
    // an apex -- `Trapezoid::corners`.
    var f = meridian_distance(cone_hard_meridian(half, r_base, r_top), q);
    let s = cone_slant(half, r_base, r_top);
    let ls = dot(s.xy, q) - s.z;
    let lm = dot(vec2<f32>(-s.x, s.y), q) - s.z;
    let lb = -q.y - half;
    let lt = q.y - half;
    f = max(f, -blend_bound(-ls, -select(lm, lb, r_base > 0.0), edge.x, edge.y));
    f = max(f, -blend_bound(-ls, -select(lm, lt, r_top > 0.0), edge.x, edge.y));
    return f;
}

fn cone_edge_normal(
    p: vec3<f32>,
    base: vec3<f32>,
    top: vec3<f32>,
    r_base: f32,
    r_top: f32,
    edge: vec2<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    let n = vec3<f32>(
        cone_edge_distance(p + vec3<f32>(h, 0.0, 0.0), base, top, r_base, r_top, edge)
            - cone_edge_distance(p - vec3<f32>(h, 0.0, 0.0), base, top, r_base, r_top, edge),
        cone_edge_distance(p + vec3<f32>(0.0, h, 0.0), base, top, r_base, r_top, edge)
            - cone_edge_distance(p - vec3<f32>(0.0, h, 0.0), base, top, r_base, r_top, edge),
        cone_edge_distance(p + vec3<f32>(0.0, 0.0, h), base, top, r_base, r_top, edge)
            - cone_edge_distance(p - vec3<f32>(0.0, 0.0, h), base, top, r_base, r_top, edge),
    );
    let l = length(n);
    if (l < 1e-12) {
        return cone_normal(p, base, top, r_base, r_top);
    }
    return n / l;
}

fn cone_distance(p: vec3<f32>, base: vec3<f32>, top: vec3<f32>, r_base: f32, r_top: f32) -> f32 {
    return axis_cone_distance(segment_local(p, base, top), r_base, r_top, distance(base, top));
}

fn cone_normal(
    p: vec3<f32>,
    base: vec3<f32>,
    top: vec3<f32>,
    r_base: f32,
    r_top: f32,
) -> vec3<f32> {
    let n = axis_cone_normal(segment_local(p, base, top), r_base, r_top, distance(base, top));
    return segment_world(n, base, top);
}

// Triply periodic minimal surfaces, matching `lattice/tpms.rs` term for term.
//
// `cell` selects the surface: 0 gyroid, 1 Schwarz P, 2 Schwarz D. It arrives as a parameter
// rather than being generated into the shader so that changing the cell costs a buffer write
// instead of a recompile, the same as every other number here.
//
// The value is divided by a bound on |grad F|, which is what makes it never overstate the
// distance to a wall. It is still not a distance — see `lattice/mod.rs` for why that is
// deliberate and what it costs.
fn tpms_f(cell: f32, p: vec3<f32>) -> f32 {
    let s = sin(p);
    let c = cos(p);
    if cell < 0.5 {
        return s.x * c.y + s.y * c.z + s.z * c.x;
    }
    if cell < 1.5 {
        return c.x + c.y + c.z;
    }
    if cell < 2.5 {
        return s.x * s.y * s.z + s.x * c.y * c.z + c.x * s.y * c.z + c.x * c.y * s.z;
    }
    // The doubled-angle terms are computed inside the branches that need them rather than
    // beside `s` and `c` above: `cell` is a uniform, so the other branches pay nothing, and
    // writing `cos(2.0 * p)` keeps this readable against `lattice/tpms.rs` term for term — which
    // is the only defence there is against a transcription error that still looks like a lattice.
    if cell < 3.5 {
        let c2 = cos(2.0 * p);
        return 2.0 * (c.x * c.y + c.y * c.z + c.z * c.x) - (c2.x + c2.y + c2.z);
    }
    if cell < 4.5 {
        return 3.0 * (c.x + c.y + c.z) + 4.0 * c.x * c.y * c.z;
    }
    let c2 = cos(2.0 * p);
    return c2.x * s.y * c.z + c.x * c2.y * s.z + s.x * c.y * c2.z;
}

fn tpms_gradient(cell: f32, p: vec3<f32>) -> vec3<f32> {
    let s = sin(p);
    let c = cos(p);
    if cell < 0.5 {
        return vec3<f32>(
            c.x * c.y - s.z * s.x,
            c.y * c.z - s.x * s.y,
            c.z * c.x - s.y * s.z,
        );
    }
    if cell < 1.5 {
        return vec3<f32>(-s.x, -s.y, -s.z);
    }
    if cell < 2.5 {
        return vec3<f32>(
            c.x * (s.y * s.z + c.y * c.z) - s.x * (s.y * c.z + c.y * s.z),
            c.y * (s.x * s.z + c.x * c.z) - s.y * (s.x * c.z + c.x * s.z),
            c.z * (s.x * s.y + c.x * c.y) - s.z * (s.x * c.y + c.x * s.y),
        );
    }
    if cell < 3.5 {
        let s2 = sin(2.0 * p);
        return vec3<f32>(
            -2.0 * s.x * (c.y + c.z) + 2.0 * s2.x,
            -2.0 * s.y * (c.z + c.x) + 2.0 * s2.y,
            -2.0 * s.z * (c.x + c.y) + 2.0 * s2.z,
        );
    }
    if cell < 4.5 {
        return vec3<f32>(
            -s.x * (3.0 + 4.0 * c.y * c.z),
            -s.y * (3.0 + 4.0 * c.z * c.x),
            -s.z * (3.0 + 4.0 * c.x * c.y),
        );
    }
    let c2 = cos(2.0 * p);
    let s2 = sin(2.0 * p);
    return vec3<f32>(
        -2.0 * s2.x * s.y * c.z - s.x * c2.y * s.z + c.x * c.y * c2.z,
        c2.x * c.y * c.z - 2.0 * s2.y * c.x * s.z - s.x * s.y * c2.z,
        -c2.x * s.y * s.z + c.x * c2.y * c.z - 2.0 * s2.z * s.x * c.y,
    );
}

// The proven bound on |grad F| at unit frequency, and the measured typical value at the
// surface. Kept beside the formulas rather than passed in, because they are properties of the
// surface and must not be able to disagree with the ones `lattice/tpms.rs` uses.
fn lattice_gradient_bound(cell: f32) -> f32 {
    // The first three are sqrt(3), each proved and each attained; the last three are bounded
    // termwise and carry slack on purpose. The reasoning is on `TpmsKind::gradient_bound`, and
    // `the_shader_divides_a_lattice_by_the_same_bound_the_cpu_does` asserts each of these
    // matches its own kind, because the same constant in two languages is exactly what drifts.
    //
    // This was a single `return` while the kinds happened to agree, and the branch is back now
    // that they do not — which is what the old comment said would have to happen.
    if cell < 2.5 {
        return 1.7320508;
    }
    if cell < 3.5 {
        return 10.392305;
    }
    if cell < 4.5 {
        return 12.124356;
    }
    if cell < 5.5 {
        return 6.9282032;
    }
    // A beam cell's field is a distance, and a distance has unit gradient by definition — so
    // there is nothing to divide away, and what is left of the division is the frequency alone.
    // This is why a beam traces several times faster than any surface above it.
    return 1.0;
}

fn tpms_surface_gradient(cell: f32) -> f32 {
    if cell < 0.5 {
        return 1.530;
    }
    if cell < 1.5 {
        return 1.321;
    }
    if cell < 2.5 {
        return 1.486;
    }
    if cell < 3.5 {
        return 4.328;
    }
    if cell < 4.5 {
        return 2.401;
    }
    return 1.759;
}

// Beam cells: struts along a wireframe, thickened to a radius.
//
// Folded into one octant first, which is exact rather than an approximation. Every cell here is
// symmetric under reflection about each mid-plane, and reflecting a point across a plane the
// strut set is symmetric in leaves its distance to that set alone — so inside the folded octant a
// single representative strut is the nearest one, and the honest minimum over 27 neighbouring
// images collapses to one or three evaluations. `lattice/beam.rs` carries the argument.
fn beam_fold(u: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.5) - abs(fract(u) - vec3<f32>(0.5));
}

fn beam_point_to_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ab = b - a;
    let denominator = dot(ab, ab);
    if denominator < 1e-18 {
        return length(p - a);
    }
    let t = clamp(dot(p - a, ab) / denominator, 0.0, 1.0);
    return length(p - (a + ab * t));
}

// Distance to the strut axes, in unit-cube coordinates.
// Diamond's struts, in the wedge its own symmetry reduces to — the same table as
// `DIAMOND_STRUTS` in `lattice/beam.rs`, and it has to stay that way. Twenty-four against six
// for the others, which is what losing the mirror fold costs; `gpu_matches_cpu`'s diamond scene
// is what says the transcription is right, because a wrong strut moves the surface.
fn beam_diamond_distance(u: vec3<f32>) -> f32 {
    // Diamond is not mirror-symmetric about the cell's mid-planes, so the fold the other cells
    // use is invalid here. It *is* invariant under permuting the coordinates, so wrap into the
    // cell and sort instead — a sixth of the cell rather than an eighth.
    let q = fract(u);
    let lo = min(min(q.x, q.y), q.z);
    let hi = max(max(q.x, q.y), q.z);
    let p = vec3<f32>(lo, q.x + q.y + q.z - lo - hi, hi);
    var segments = array<vec3<f32>, 48>(
        vec3<f32>(0.00, 0.00, 0.00), vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(0.00, 0.00, 1.00), vec3<f32>(-0.25, 0.25, 0.75),
        vec3<f32>(0.00, 0.00, 1.00), vec3<f32>(0.25, -0.25, 0.75),
        vec3<f32>(0.00, 0.00, 1.00), vec3<f32>(0.25, 0.25, 1.25),
        vec3<f32>(0.00, 0.50, 0.50), vec3<f32>(-0.25, 0.25, 0.75),
        vec3<f32>(0.00, 0.50, 0.50), vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(0.00, 0.50, 0.50), vec3<f32>(0.25, 0.75, 0.75),
        vec3<f32>(0.00, 1.00, 1.00), vec3<f32>(0.25, 0.75, 0.75),
        vec3<f32>(0.50, 0.00, 0.50), vec3<f32>(0.25, -0.25, 0.75),
        vec3<f32>(0.50, 0.00, 0.50), vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(0.50, 0.00, 0.50), vec3<f32>(0.75, 0.25, 0.75),
        vec3<f32>(0.50, 0.50, 0.00), vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(0.50, 0.50, 1.00), vec3<f32>(0.25, 0.25, 1.25),
        vec3<f32>(0.50, 0.50, 1.00), vec3<f32>(0.25, 0.75, 0.75),
        vec3<f32>(0.50, 0.50, 1.00), vec3<f32>(0.75, 0.25, 0.75),
        vec3<f32>(0.50, 0.50, 1.00), vec3<f32>(0.75, 0.75, 1.25),
        vec3<f32>(0.50, 1.00, 0.50), vec3<f32>(0.25, 0.75, 0.75),
        vec3<f32>(0.50, 1.00, 0.50), vec3<f32>(0.75, 0.75, 0.25),
        vec3<f32>(0.50, 1.00, 0.50), vec3<f32>(0.75, 1.25, 0.75),
        vec3<f32>(1.00, 0.50, 0.50), vec3<f32>(0.75, 0.25, 0.75),
        vec3<f32>(1.00, 0.50, 0.50), vec3<f32>(0.75, 0.75, 0.25),
        vec3<f32>(1.00, 1.00, 1.00), vec3<f32>(0.75, 0.75, 1.25),
        vec3<f32>(1.00, 1.00, 1.00), vec3<f32>(0.75, 1.25, 0.75),
        vec3<f32>(1.00, 1.00, 1.00), vec3<f32>(1.25, 0.75, 0.75)
    );
    var best = 1e30;
    for (var i = 0u; i < 24u; i = i + 1u) {
        best = min(best, beam_point_to_segment(p, segments[2u * i], segments[2u * i + 1u]));
    }
    return best;
}

fn beam_axis_distance(cell: f32, u: vec3<f32>) -> f32 {
    if cell > 9.5 {
        return beam_diamond_distance(u);
    }
    let f = beam_fold(u);
    if cell < 6.5 {
        // Cubic: the nearest axis line runs through the folded origin, so the distance to it is
        // the distance in the other two coordinates.
        return min(min(length(f.yz), length(f.zx)), length(f.xy));
    }
    if cell < 7.5 {
        // BCC: every strut runs corner to centre, and the main diagonal is the only one inside
        // the folded octant.
        return beam_point_to_segment(f, vec3<f32>(0.0), vec3<f32>(0.5));
    }
    let o = vec3<f32>(0.0);
    if cell < 8.5 {
        // FCC: three face diagonals and three cube edges, all from the folded origin.
        return min(
            min(
                min(
                    beam_point_to_segment(f, o, vec3<f32>(0.0, 1.0, 1.0)),
                    beam_point_to_segment(f, o, vec3<f32>(1.0, 0.0, 1.0)),
                ),
                min(
                    beam_point_to_segment(f, o, vec3<f32>(1.0, 1.0, 0.0)),
                    beam_point_to_segment(f, o, vec3<f32>(1.0, 0.0, 0.0)),
                ),
            ),
            min(
                beam_point_to_segment(f, o, vec3<f32>(0.0, 1.0, 0.0)),
                beam_point_to_segment(f, o, vec3<f32>(0.0, 0.0, 1.0)),
            ),
        );
    }
    // Octet (and nothing below it: diamond returned above, before the fold that would have been
    // wrong for it). The three struts from the folded corner to the face centres around it, and
    // the three octahedron edges joining those face centres.
    let fa = vec3<f32>(0.5, 0.5, 0.0);
    let fb = vec3<f32>(0.5, 0.0, 0.5);
    let fc = vec3<f32>(0.0, 0.5, 0.5);
    return min(
        min(
            min(beam_point_to_segment(f, o, fa), beam_point_to_segment(f, o, fb)),
            min(beam_point_to_segment(f, o, fc), beam_point_to_segment(f, fa, fb)),
        ),
        min(beam_point_to_segment(f, fa, fc), beam_point_to_segment(f, fb, fc)),
    );
}

// The cell map: where a point sits inside the tiling pattern.
//
// `map` is 0 for rectangular and 1 for cylindrical. The three parameters that follow mean
// different things under each — one cell size, or a radial size, an axial size and a count —
// which is exactly the asymmetry the split exists to allow. The unit cell's formula is untouched
// by which map is in front of it.
fn lattice_args(p: vec3<f32>, map: f32, a: f32, b: f32, c: f32) -> vec3<f32> {
    if map < 0.5 {
        return p * (6.2831853 / max(a, 1e-9));
    }
    if map < 1.5 {
        let r = length(p.xy);
        // `atan2(0, 0)` is defined here, so the axis itself is safe; the division that is not
        // safe lives in `lattice_frequency`, which guards its own.
        let theta = atan2(p.y, p.x);
        return vec3<f32>(6.2831853 * r / max(a, 1e-9), c * theta, 6.2831853 * p.z / max(b, 1e-9));
    }
    // Spherical: radius, azimuth, polar angle from +Z.
    let rr = length(p);
    let azimuth = atan2(p.y, p.x);
    let polar = acos(clamp(select(0.0, p.z / rr, rr > 1e-9), -1.0, 1.0));
    return vec3<f32>(6.2831853 * rr / max(a, 1e-9), b * azimuth, c * polar);
}

// How fast each cell argument advances per millimetre along its own direction.
//
// Constant for a rectangular map. Cylindrical's circumferential term is `N/r`, which is the
// whole reason wall thickness cannot be hoisted out of the loop: the pattern is compressed near
// the axis and stretched away from it, so one isovalue would be right at a single radius.
fn lattice_frequency(p: vec3<f32>, map: f32, a: f32, b: f32, c: f32) -> vec3<f32> {
    if map < 0.5 {
        let w = 6.2831853 / max(a, 1e-9);
        return vec3<f32>(w, w, w);
    }
    if map < 1.5 {
        let r = max(length(p.xy), 1e-9);
        return vec3<f32>(6.2831853 / max(a, 1e-9), c / r, 6.2831853 / max(b, 1e-9));
    }
    // The azimuth is measured on a circle of radius `r·sin(polar)`, which vanishes at a pole
    // while `r` does not — so its frequency rises there the way the cylindrical one rises at
    // the axis. Missing that thins the walls toward the poles.
    let rr = max(length(p), 1e-9);
    let rho = max(length(p.xy), 1e-9);
    return vec3<f32>(6.2831853 / max(a, 1e-9), b / rho, c / rr);
}

// The largest frequency the map presents at a stated distance from its axis or centre.
//
// The building block of a bound that is *local* rather than global. One constant valid
// everywhere has to be taken where the cells collapse, and out at the rim of a real part that
// understated the available room by more than a hundredfold — which costs a ray that many more
// steps than it needs, and shows as speckle wherever one runs out of them.
fn lattice_frequency_at(map: f32, a: f32, b: f32, c: f32, r: f32) -> f32 {
    let rr = max(r, 1e-9);
    if map < 0.5 {
        return 6.2831853 / max(a, 1e-9);
    }
    if map < 1.5 {
        return max(max(6.2831853 / max(a, 1e-9), c / rr), 6.2831853 / max(b, 1e-9));
    }
    return max(max(6.2831853 / max(a, 1e-9), b / rr), c / rr);
}

// Where a map's cells have closed up, inside which the field is solid.
//
// Takes a feature size rather than a wall thickness, because a solid-phase lattice has no wall
// and still has to answer this. `lattice_feature_size` is what supplies it for both phases.
fn lattice_degenerate_radius(map: f32, b: f32, c: f32, feature: f32) -> f32 {
    if map < 0.5 {
        return 0.0;
    }
    return select(c, max(b, c), map >= 1.5) * feature / 6.2831853;
}

// The isovalue at which `F <= iso` fills the requested fraction of a cell.
//
// **The same table as `TpmsKind::density_isovalues` on the CPU, and it has to stay that way** —
// `the_shader_reads_a_density_through_the_same_table_the_cpu_does` reads both and compares them,
// because a table in two languages is exactly what drifts. Measured over one period by
// `tpms_profile.rs`; there is no closed form for any of these surfaces to derive it from.
//
// Transcribed rather than converted CPU-side on purpose: a density can be driven by a named
// parameter, and folding it to an isovalue during codegen would freeze it into the shader.
fn tpms_density_entry(cell: f32, i: u32) -> f32 {
    // `var` rather than `const` so the index may be dynamic, which is what the interpolation
    // below needs.
    var gyroid = array<f32, 11>(
        -1.500, -1.197, -0.913, -0.615, -0.310, 0.000, 0.310, 0.615, 0.913, 1.197, 1.500
    );
    var schwarz_p = array<f32, 11>(
        -3.000, -1.621, -1.051, -0.700, -0.350, 0.000, 0.350, 0.700, 1.051, 1.621, 3.000
    );
    var schwarz_d = array<f32, 11>(
        -1.414, -0.956, -0.726, -0.485, -0.243, 0.000, 0.243, 0.485, 0.726, 0.956, 1.414
    );
    // Schoen IWP is the one that is not antisymmetric — it runs from -5 to +3 and its half-way
    // entry is 0.232, not zero. Reading it through a symmetric assumption would give the wrong
    // amount of material and nothing about the result would look wrong.
    var schoen_iwp = array<f32, 11>(
        -5.000, -3.002, -2.164, -1.339, -0.538, 0.232, 0.954, 1.618, 2.205, 2.691, 3.000
    );
    var neovius = array<f32, 11>(
        -13.000, -4.857, -2.301, -0.980, -0.456, 0.000, 0.456, 0.980, 2.301, 4.857, 13.000
    );
    var fischer_koch_s = array<f32, 11>(
        -1.414, -0.784, -0.598, -0.404, -0.203, 0.000, 0.203, 0.404, 0.598, 0.784, 1.414
    );
    if cell < 0.5 {
        return gyroid[i];
    }
    if cell < 1.5 {
        return schwarz_p[i];
    }
    if cell < 2.5 {
        return schwarz_d[i];
    }
    if cell < 3.5 {
        return schoen_iwp[i];
    }
    if cell < 4.5 {
        return neovius[i];
    }
    return fischer_koch_s[i];
}

fn tpms_density_isovalue(cell: f32, density: f32) -> f32 {
    let d = clamp(density, 0.0, 1.0);
    let x = d * 10.0;
    let i = min(u32(floor(x)), 9u);
    return mix(tpms_density_entry(cell, i), tpms_density_entry(cell, i + 1u), x - f32(i));
}

// The smallest cell dimension that is a fixed length rather than a function of radius.
//
// A curved map has one extent that shrinks toward its axis — the circumferential one, and the
// whole reason `lattice_degenerate_radius` exists — and one or two that do not. This is the
// smallest of the ones that do not, which is the only cell size a solid can measure a strut
// against without asking where it is.
fn lattice_smallest_cell_extent(map: f32, a: f32, b: f32, c: f32) -> f32 {
    if map < 1.5 && map >= 0.5 {
        return min(a, b);
    }
    return a;
}

// The smallest material feature the lattice has, in millimetres.
//
// A wall has a thickness and there is nothing to derive. A solid's material is a fixed *fraction*
// of each cell, so it never reaches a fixed width — `min(d, 1-d)` of the smallest fixed cell
// extent is the estimate, the `min` because the thin feature is the strut at low density and the
// void at high. `TpmsLattice::feature_size_mm` carries why an estimate is allowed to stand here
// and what each direction of error costs; this must agree with it.
fn lattice_feature_size(map: f32, a: f32, b: f32, c: f32, phase: f32, sizing: f32) -> f32 {
    if phase < 0.5 {
        return sizing;
    }
    if phase > 1.5 {
        // A strut's smallest feature is its diameter, exactly — the one arm here with nothing
        // estimated in it.
        return 2.0 * sizing;
    }
    let d = clamp(sizing, 0.0, 1.0);
    return min(d, 1.0 - d) * lattice_smallest_cell_extent(map, a, b, c);
}

// How far a point is from the axis or centre, in whichever sense this map's cells close up.
fn lattice_radius_of(p: vec3<f32>, map: f32) -> f32 {
    if map < 0.5 {
        return 3.4e38;
    }
    return select(length(p.xy), length(p), map >= 1.5);
}

// `sizing` is the one number the phase carries: a wall thickness in millimetres under sheet, a
// relative density under solid. They are different quantities and neither converts into the
// other, which is why one slot serves both rather than two slots one of which is always dormant.
// Three 32-bit values from an integer cell index and a seed.
//
// **PCG3D**, Jarzynski and Olano, JCGT 9(3) 2020, table 1 -- the same arithmetic as
// `fieldform_field::voronoi_hash3`, which is pinned to a table of known values by
// `a_hash_is_the_same_number_wherever_it_is_computed`. Every operation is `u32` and wraps on
// both sides, so this is a transcription rather than an approximation. The usual
// `sin(dot(p, k)) * 43758.5453` is disqualified: it is not reproducible across drivers, and D7
// requires that an arrangement be a function of what was saved.
fn voronoi_hash3(c: vec3<i32>, seed: u32) -> vec3<u32> {
    var v = vec3<u32>(
        bitcast<u32>(c.x) ^ seed,
        bitcast<u32>(c.y) ^ (seed * 0x9E3779B9u),
        bitcast<u32>(c.z) ^ (seed * 0x85EBCA6Bu),
    );
    v = v * 1664525u + vec3<u32>(1013904223u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// A cell index folded into `0..n`, the way the field's `rem_euclid` folds it -- and written
// without `%`, deliberately.
//
// **WGSL's `%` was not usable here.** With a negative left operand it did not agree with
// `i32::rem_euclid` on this driver: every cell below nought on the wrapping axis drew a
// different seed from the one the field drew, so half of every cylindrical Voronoi lattice was
// a different arrangement on the two sides. The symptom was a 34 mm surface residual in
// `gpu_matches_cpu` with the hash itself provably identical, and `voronoi_probe` is what
// separated the two -- the rectangular rows agreed to 4e-7 at every jitter while the wrapped
// ones parted on exactly the points whose cell index was negative.
//
// The float division is exact for any index a lattice can reach; the two guards cover the
// rounding at an exact multiple.
fn cell_index_wrap(i: i32, n: i32) -> i32 {
    var m = i - n * i32(floor(f32(i) / f32(n)));
    if m < 0 {
        m = m + n;
    }
    if m >= n {
        m = m - n;
    }
    return m;
}

// The three nearest seeds and where they are, over the `(2R+1)^3` cells about `u`.
//
// `period` wraps the hash index on the axis a curved map closes on, leaving the seed's position
// where it is. Without it the cell below nought and the cell below the count draw different
// seeds although they are the same place on a cylinder, and the field jumps along the seam --
// which is a ray stepping through a wall, not merely a mark on the surface. `nearest_seeds` is
// the field's side of this.
struct VoronoiRing {
    d: vec3<f32>,
    near: vec3<f32>,
    second: vec3<f32>,
    third: vec3<f32>,
}

fn voronoi_ring(u: vec3<f32>, jitter: f32, seed: u32, period: f32, radius: i32) -> VoronoiRing {
    var out: VoronoiRing;
    out.d = vec3<f32>(3.4e38, 3.4e38, 3.4e38);
    out.near = vec3<f32>(0.0);
    out.second = vec3<f32>(0.0);
    out.third = vec3<f32>(0.0);
    let base = vec3<i32>(floor(u));
    let n = i32(period);
    for (var dz = -radius; dz <= radius; dz = dz + 1) {
        for (var dy = -radius; dy <= radius; dy = dy + 1) {
            for (var dx = -radius; dx <= radius; dx = dx + 1) {
                let c = base + vec3<i32>(dx, dy, dz);
                var index = c;
                if period > 0.5 {
                    index.y = cell_index_wrap(c.y, n);
                }
                let h = voronoi_hash3(index, seed);
                let unit = vec3<f32>(f32(h.x), f32(h.y), f32(h.z)) * 2.3283064e-10;
                let at = vec3<f32>(c) + vec3<f32>(0.5) + jitter * (unit - vec3<f32>(0.5));
                let dist = length(u - at);
                if dist < out.d.x {
                    out.d = vec3<f32>(dist, out.d.x, out.d.y);
                    out.third = out.second;
                    out.second = out.near;
                    out.near = at;
                } else if dist < out.d.y {
                    out.d = vec3<f32>(out.d.x, dist, out.d.y);
                    out.third = out.second;
                    out.second = at;
                } else if dist < out.d.z {
                    out.d.z = dist;
                    out.third = at;
                }
            }
        }
    }
    return out;
}

// `(min(d_k, l_R) - d_1) / 2` and its gradient, both in cell units.
//
// **The ring radius is a correctness parameter and the clamp is what makes it one.** A seed in a
// cell at Chebyshev distance `R+1` or more is at least `l_R = R + (1 - jitter)/2` away, so
// clamping `d_k` to `l_R` can only understate -- the safe direction. But if `d_1` itself exceeds
// `l_R` the true nearest seed may not have been found at all, nothing may be concluded, and the
// ring widens. `R = 2` needs no third pass: `l_2 >= 2` and the own-cell seed is at most
// `sqrt(3) < 2`. `voronoi_measure` is the field's side, with the derivation.
struct VoronoiMeasure {
    value: f32,
    gradient: vec3<f32>,
}

fn voronoi_measure(order: u32, u: vec3<f32>, raw_jitter: f32, seed: u32, period: f32) -> VoronoiMeasure {
    // **Clamped here, as `Lattice::with_scatter` clamps it there**, and for more than agreement:
    // above one a seed leaves its own cell, and then a ring search can miss the true nearest and
    // the whole bound collapses. The field clamps where the document's value is installed, which
    // is a place the shader has no equivalent of -- a jitter travels here as a params slot, and a
    // slot can carry whatever an expression parameter evaluated to.
    let jitter = clamp(raw_jitter, 0.0, 1.0);
    // One ring and one ceiling, never widened. `voronoi_measure` on the field's side carries the
    // whole argument, including why §4.3's own widen-and-repeat is unsound: the ceiling is
    // `R + (1 - jitter)/2`, so widening raises it by a cell and the value steps upward across
    // the surface where `d_1` reaches it.
    let ceiling = 1.0 + (1.0 - jitter) * 0.5;
    let hit = voronoi_ring(u, jitter, seed, period, 1);
    var far_d = hit.d.y;
    var far_at = hit.second;
    if order > 1u {
        far_d = hit.d.z;
        far_at = hit.third;
    }
    let reach = min(far_d, ceiling);
    var out: VoronoiMeasure;
    out.value = (reach - hit.d.x) * 0.5;
    out.gradient = vec3<f32>(0.0);
    if out.value <= 0.0 {
        out.value = 0.0;
        return out;
    }
    var away_near = vec3<f32>(0.0);
    if hit.d.x > 1e-9 {
        away_near = (u - hit.near) / hit.d.x;
    }
    var away_far = vec3<f32>(0.0);
    if far_d > 1e-9 {
        away_far = (u - far_at) / far_d;
    }
    // A clamped term contributes no gradient, since it is a constant there.
    out.gradient = select((away_far - away_near) * 0.5, -away_near * 0.5, far_d > ceiling);
    return out;
}

// Which axis a map closes on, as a whole number of cells, or nought where none does.
fn lattice_wrap_period(map: f32, b: f32, c: f32) -> f32 {
    if map < 0.5 {
        return 0.0;
    }
    if map < 1.5 {
        return c;
    }
    return b;
}

// Distance to a honeycomb's nearest wall mid-plane, `across` being the hexagon's flat-to-flat
// width. `PrismKind::wall_distance` is the same arithmetic on the other side and has to stay
// the same: the nearest of two interleaved lattices of centres names the hexagon the point is
// in, and inside a convex polygon the distance to its boundary is the distance to its nearest
// edge -- so this is exact, and a honeycomb's wall is as thick as it was asked to be.
fn honeycomb_wall_distance(q: vec2<f32>, across: f32) -> f32 {
    let period = vec2<f32>(1.7320508 * across, across);
    let a = q - round(q / period) * period;
    let b = q - (round(q / period - vec2<f32>(0.5)) + vec2<f32>(0.5)) * period;
    var c = a;
    if dot(b, b) < dot(a, a) {
        c = b;
    }
    let r = across * 0.5;
    let k = vec2<f32>(-0.8660254, 0.5);
    var h = abs(c);
    h = h - 2.0 * min(dot(k, h), 0.0) * k;
    let half_edge = r * 0.57735027;
    let d = vec2<f32>(h.x - clamp(h.x, -half_edge, half_edge), h.y - r);
    // Negative inside the hexagon, so the distance to the wall is its size.
    if d.y < 0.0 {
        return length(d);
    }
    return -length(d);
}

// `shift` is a sheet's mid-surface offset in millimetres: the wall is banded about `F = c`
// rather than about the cell's own surface, with `c` converted through the same local scale the
// thickness is, so the two passages it separates can differ. Nought is a plain sheet, and it is
// nought for every other phase. `Lattice::midsurface_at` is the field's side of it.
fn lattice_distance(
    p: vec3<f32>, cell: f32, map: f32, a: f32, b: f32, c: f32, phase: f32, sizing: f32,
    shift: f32, jitter: f32, seed_lo: f32, seed_hi: f32
) -> f32 {
    let seed = u32(seed_lo) | (u32(seed_hi) << 16u);
    if cell > 10.5 && cell < 11.5 {
        // A prism: a planar tiling carried through the third axis, measured in millimetres
        // with no cell coordinates at all. Rectangular maps only -- both sides refuse the rest
        // by name -- so `a` is the cell size typed, the hexagon's flat-to-flat width.
        return honeycomb_wall_distance(p.xy, a) - sizing * 0.5;
    }
    let feature = lattice_feature_size(map, a, b, c, phase, sizing);
    // Inside the radius where cells have collapsed to the size of the material in them,
    // neighbouring material has merged and the answer is solid — which is what the arithmetic
    // converges to anyway, but it converges through arbitrarily large isovalues, so it is
    // stated instead.
    if map >= 0.5 {
        let r_min = lattice_degenerate_radius(map, b, c, feature);
        if lattice_radius_of(p, map) < r_min {
            return -min(r_min - lattice_radius_of(p, map), feature * 0.5);
        }
    }
    let args = lattice_args(p, map, a, b, c);
    let w = lattice_frequency(p, map, a, b, c);
    if cell > 11.5 {
        // A difference of order statistics over a jittered seed grid, in cell units, converted
        // the way a beam's distance is: back into argument units and divided by the largest
        // local frequency, in the two steps that keep a curved map sound. What is subtracted is
        // a length in millimetres -- half a wall for the foam, a whole radius for the beam --
        // so there is no isovalue here and no calibration: the measure understates, so the wall
        // comes out at least as thick as typed. `StochasticKind::wall_spread` measures by how
        // much, and the panel says so where the number is entered.
        let order = select(1u, 2u, cell > 12.5);
        let period = lattice_wrap_period(map, b, c);
        let m = voronoi_measure(order, args / 6.2831853, jitter, seed, period);
        let raw_v = m.value * 6.2831853;
        let r_v = lattice_radius_of(p, map);
        let first_v = lattice_frequency_at(map, a, b, c, r_v);
        let floor_v = lattice_degenerate_radius(map, b, c, feature);
        let bound_v =
            lattice_frequency_at(map, a, b, c, max(r_v - abs(raw_v / first_v), floor_v));
        let size_v = select(sizing, sizing * 0.5, phase < 0.5);
        return raw_v / bound_v - size_v;
    }
    if cell > 5.5 {
        // A distance, not a banded value. The struts are axes in cell space, so the distance to
        // them is converted into millimetres and the radius taken off it — with nothing to band
        // and nothing averaged on the way. Dividing by the largest local frequency is what makes
        // the conversion safe where a map scales the axes differently, and it is exact where
        // they agree.
        let to_axis = beam_axis_distance(cell, args / 6.2831853) * 6.2831853;
        let gb = lattice_gradient_bound(cell);
        let r_beam = lattice_radius_of(p, map);
        let first_beam = lattice_frequency_at(map, a, b, c, r_beam) * gb;
        let floor_beam = lattice_degenerate_radius(map, b, c, feature);
        let reach_beam = abs(to_axis / first_beam);
        let bound_beam =
            lattice_frequency_at(map, a, b, c, max(r_beam - reach_beam, floor_beam)) * gb;
        return to_axis / bound_beam - sizing;
    }
    var raw: f32;
    if phase < 0.5 {
        // Per-argument gradient: the cell's whole-vector magnitude split evenly across its three
        // arguments, which is exact for these three surfaces by their symmetry. With every
        // frequency equal this collapses to `thickness/2 * omega * g`, the rectangular formula.
        let g = tpms_surface_gradient(cell) * 0.5773503;
        let wg = w * g;
        let iso = sizing * 0.5 * length(wg);
        raw = abs(tpms_f(cell, args) - shift * length(wg)) - iso;
    } else {
        // No frequency anywhere in this branch, and that is the point: a density is a fraction
        // of a cell rather than a length, so squeezing the cells leaves it alone. Everything
        // above about the Jacobian is about converting millimetres, and there are none here.
        raw = tpms_f(cell, args) - tpms_density_isovalue(cell, sizing);
    }

    // A local bound, made safe by asking twice — the same two steps the CPU takes, and it has
    // to be the same or the two paths put the surface in one place and scale the field around
    // it differently. A first estimate uses the bound where the point is; a ray may travel that
    // far inward, so the second takes the bound there, where the frequency is higher and the
    // answer is therefore smaller. Never nearer than the radius at which the field goes solid.
    let gb = lattice_gradient_bound(cell);
    let r = lattice_radius_of(p, map);
    let first = lattice_frequency_at(map, a, b, c, r) * gb;
    let floor_r = lattice_degenerate_radius(map, b, c, feature);
    let reach = abs(raw / first);
    let bound = lattice_frequency_at(map, a, b, c, max(r - reach, floor_r)) * gb;
    return raw / bound;
}

fn lattice_normal(
    p: vec3<f32>, cell: f32, map: f32, a: f32, b: f32, c: f32, phase: f32, sizing: f32,
    shift: f32, jitter: f32, seed_lo: f32, seed_hi: f32
) -> vec3<f32> {
    let seed = u32(seed_lo) | (u32(seed_hi) << 16u);
    if cell > 10.5 && cell < 11.5 {
        // Away from the nearest wall, in the plane: a prism has no gradient along its axis.
        let h = 1e-3;
        let g = vec2<f32>(
            honeycomb_wall_distance(p.xy + vec2<f32>(h, 0.0), a)
                - honeycomb_wall_distance(p.xy - vec2<f32>(h, 0.0), a),
            honeycomb_wall_distance(p.xy + vec2<f32>(0.0, h), a)
                - honeycomb_wall_distance(p.xy - vec2<f32>(0.0, h), a),
        );
        let len = length(g);
        if len < 1e-9 {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
        return vec3<f32>(g / len, 0.0);
    }
    let args = lattice_args(p, map, a, b, c);
    let w = lattice_frequency(p, map, a, b, c);
    // The wall is where |F| is small, so its outward direction is the gradient of |F| — the
    // gradient of F turned around wherever F is negative. Taken in argument space and carried
    // back through the map's own scaling, which is what makes it right under a map that
    // stretches one direction differently from another.
    //
    // **A solid does not get the flip.** A wall has two sides, which is what the sign tracks; a
    // labyrinth has one, and its outward direction is the gradient unturned. Flipping it anyway
    // shades half the surface inside-out, which reads as a lighting bug a long way from here.
    var g = tpms_gradient(cell, args) * w;
    if cell > 11.5 {
        // Analytic rather than a central difference, and that is a cost decision as much as an
        // accuracy one: a difference would be six more ring passes, and the pass already returns
        // both seeds the gradient is made of.
        let order_n = select(1u, 2u, cell > 12.5);
        let m = voronoi_measure(
            order_n, args / 6.2831853, jitter, seed, lattice_wrap_period(map, b, c)
        );
        g = m.gradient * w;
    } else if cell > 5.5 {
        // A central difference of the strut distance in cell space. Analytic would mean
        // differentiating a `min` over segments — the same expression with the branch written
        // out, and one more place for a sign to be wrong. The 1/2h drops out under the
        // normalisation below, so it is not written.
        let h = 1e-3;
        let uu = args / 6.2831853;
        g = vec3<f32>(
            beam_axis_distance(cell, uu + vec3<f32>(h, 0.0, 0.0))
                - beam_axis_distance(cell, uu - vec3<f32>(h, 0.0, 0.0)),
            beam_axis_distance(cell, uu + vec3<f32>(0.0, h, 0.0))
                - beam_axis_distance(cell, uu - vec3<f32>(0.0, h, 0.0)),
            beam_axis_distance(cell, uu + vec3<f32>(0.0, 0.0, h))
                - beam_axis_distance(cell, uu - vec3<f32>(0.0, 0.0, h)),
        ) * w;
    } else if phase < 0.5 {
        // Turned about the wall's own middle, which is where a walled sheet's two faces part.
        let gs = tpms_surface_gradient(cell) * 0.5773503;
        g = g * sign(tpms_f(cell, args) - shift * length(w * gs));
    }
    if map < 0.5 {
        let len = length(g);
        if len < 1e-9 {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
        return g / len;
    }
    // The gradient arrives in the map's own basis and has to be turned back into the frame's,
    // which depends on where the point is.
    let rho = length(p.xy);
    if rho < 1e-9 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let cs = p.x / rho;
    let sn = p.y / rho;
    var world: vec3<f32>;
    if map < 1.5 {
        world = vec3<f32>(cs, sn, 0.0) * g.x + vec3<f32>(-sn, cs, 0.0) * g.y
            + vec3<f32>(0.0, 0.0, 1.0) * g.z;
    } else {
        let rr = max(length(p), 1e-9);
        let sin_polar = rho / rr;
        let cos_polar = p.z / rr;
        world = vec3<f32>(cs * sin_polar, sn * sin_polar, cos_polar) * g.x
            + vec3<f32>(-sn, cs, 0.0) * g.y
            + vec3<f32>(cs * cos_polar, sn * cos_polar, -sin_polar) * g.z;
    }
    let len = length(world);
    if len < 1e-9 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return world / len;
}

// A half-space, facing +Z: the signed height above the plane. Unit gradient by inspection,
// and the only leaf here whose surface is not confined to any box.
// `facing` is +1 filling below the plane and -1 filling above it, passed as a parameter so
// flipping one costs a buffer write rather than a shader recompile.
fn plane_distance(p: vec3<f32>, origin: vec3<f32>, facing: f32) -> f32 {
    return (p.z - origin.z) * facing;
}

fn plane_normal(p: vec3<f32>, origin: vec3<f32>, facing: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, facing);
}

// A single point, as the distance to it. Encloses no volume, so it never shows a surface; it
// is here to be measured from.
fn point_distance(p: vec3<f32>, at: vec3<f32>) -> f32 {
    return length(p - at);
}

fn point_normal(p: vec3<f32>, at: vec3<f32>) -> vec3<f32> {
    let offset = p - at;
    let len = length(offset);
    if len < 1e-9 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return offset / len;
}

fn torus_distance(p: vec3<f32>, centre: vec3<f32>, major: f32, minor: f32) -> f32 {
    let q = p - centre;
    return length(vec2<f32>(length(q.xy) - major, q.z)) - minor;
}

fn torus_normal(p: vec3<f32>, centre: vec3<f32>, major: f32, minor: f32) -> vec3<f32> {
    let q = p - centre;
    let ring = length(q.xy) - major;
    let len = max(length(vec2<f32>(ring, q.z)), 1e-9);
    return radial_direction(q) * (ring / len) + vec3<f32>(0.0, 0.0, q.z / len);
}

fn capsule_distance(
    p: vec3<f32>,
    start: vec3<f32>,
    end: vec3<f32>,
    r_start: f32,
    r_end: f32,
) -> f32 {
    return axis_capsule_distance(
        segment_local(p, start, end),
        r_start,
        r_end,
        distance(start, end),
    );
}

fn capsule_normal(
    p: vec3<f32>,
    start: vec3<f32>,
    end: vec3<f32>,
    r_start: f32,
    r_end: f32,
) -> vec3<f32> {
    let n = axis_capsule_normal(
        segment_local(p, start, end),
        r_start,
        r_end,
        distance(start, end),
    );
    return segment_world(n, start, end);
}

// Signed distance to an axis-aligned box, used for the conservative answer outside a bake.
fn box_distance(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let d = max(lo - p, p - hi);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// The distance to a body's guard box, whose six corners sit in `params` from `slot`: lo then
// hi. A function here rather than six reads spelled out per body in the generated fold, so a
// thousand bodies add a thousand short lines rather than a thousand long ones -- see
// `body_box_slots` in codegen.rs for what the box is for.
fn box_guard(p: vec3<f32>, slot: u32) -> f32 {
    let lo = vec3<f32>(params[slot], params[slot + 1u], params[slot + 2u]);
    let hi = vec3<f32>(params[slot + 3u], params[slot + 4u], params[slot + 5u]);
    return box_distance(p, lo, hi);
}

// Signed distance to a closed polyline, in the plane it was uploaded in.
//
// The same field `Profile::distance` computes on the other side, and it has to stay the same:
// the magnitude is the minimum over the segments and the sign is crossing parity, counted with
// the same half-open rule so a ray through a vertex is counted once rather than twice. The
// points are already in the block's own frame — `CurveUse::Profile` says why they are
// pre-transformed rather than turned here.
//
// Even-odd, deliberately. A profile that crosses itself has no inside until a rule is picked,
// and this is the one drawing programs use.
fn profile_distance(index: u32, q: vec2<f32>) -> f32 {
    let span = curve_spans[index];
    // A baked grid answers in one read where the walk below reads every segment of the
    // boundary. See `baked_profile.rs` for why it never overstates and what the margin is. A
    // ray finishing on the outline (`exact_reads`) reads it from the block's own few segments
    // where it is near enough for those to be the answer, and the grid where it is not.
    if (span.bake_wide > 0u) {
        if (exact_reads) {
            let local = outline_distance_local(span, q);
            if (local.x > 0.5) {
                return local.y;
            }
        }
        return baked_profile_distance(span, q);
    }
    if (span.count < 3u) {
        return 1e30;
    }
    var nearest = 1e30;
    var inside = false;
    for (var i = 0u; i < span.count; i = i + 1u) {
        let a = curve_points[span.start + i].xy;
        let j = select(i + 1u, 0u, i + 1u == span.count);
        let b = curve_points[span.start + j].xy;

        let ab = b - a;
        let len2 = dot(ab, ab);
        var t = 0.0;
        if (len2 > 0.0) {
            t = clamp(dot(q - a, ab) / len2, 0.0, 1.0);
        }
        nearest = min(nearest, length(q - (a + ab * t)));

        if ((a.y > q.y) != (b.y > q.y)) {
            let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if (q.x < x) {
                inside = !inside;
            }
        }
    }
    return select(nearest, -nearest, inside);
}

// The section as the normal pass reads it: the bake, and nothing else where every section has
// one. See the profile-solids marker below for why the normal pass has a reader of its own: what
// the value pass walks is what the driver's compile pays for, and a differenced normal does not
// need the exact finish. A section the tolerance could not bake (`bake_cells`) has nothing here
// but the walk; the renderer keeps the walk's line only in a document holding such a section,
// and cuts it to `return 1e30;` otherwise, so the normal pass of every other document compiles
// without a loop in it. That edit is the one place the source the driver sees differs from the
// generator's, and the renderer's pipeline cache compares the edited text.
fn profile_distance_bake(index: u32, q: vec2<f32>) -> f32 {
    let span = curve_spans[index];
    if (span.bake_wide > 0u) {
        return baked_profile_distance(span, q);
    }
    return profile_distance(index, q); // unbaked-section-walk
}

// The section offset outward by `t` with its corners **mitred**: the same walk, over a
// boundary whose every vertex has moved by `t` times its own miter vector.
//
// `miters` is a span of the same length as `index`'s, in step with it -- see
// `CurveUse::Miters`. The arithmetic is `Profile::offset_by`'s, line for line, which is what
// lets a drafted solid be the same shape on both sides.
//
// **No bake here, and none is wanted.** A grid holds the section as drawn; the rounded offset
// is that reading less a number, which a grid can answer, and a mitre is a different boundary,
// which it cannot. The field walks for a sharp draft too, so the two agree.
fn profile_mitred(index: u32, miters: u32, q: vec2<f32>, t: f32) -> f32 {
    let span = curve_spans[index];
    let mspan = curve_spans[miters];
    if (span.count < 3u || mspan.count < span.count) {
        return 1e30;
    }
    var nearest = 1e30;
    var inside = false;
    for (var i = 0u; i < span.count; i = i + 1u) {
        let j = select(i + 1u, 0u, i + 1u == span.count);
        let a = curve_points[span.start + i].xy + curve_points[mspan.start + i].xy * t;
        let b = curve_points[span.start + j].xy + curve_points[mspan.start + j].xy * t;

        let ab = b - a;
        let len2 = dot(ab, ab);
        var s = 0.0;
        if (len2 > 0.0) {
            s = clamp(dot(q - a, ab) / len2, 0.0, 1.0);
        }
        nearest = min(nearest, length(q - (a + ab * s)));

        if ((a.y > q.y) != (b.y > q.y)) {
            let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if (q.x < x) {
                inside = !inside;
            }
        }
    }
    return select(nearest, -nearest, inside);
}

// A wall round a loop, offset outward by `t` with its corners mitred: the loop's own mitred
// reading, folded about the band exactly as `wall_distance` folds the plain one. An open wall
// never reaches here -- a draft on one is refused, having no inside to be pushed out from.
fn wall_mitred(chain: u32, miters: u32, q: vec2<f32>, t: f32, thickness: f32, facing: f32) -> f32 {
    let half = 0.5 * thickness;
    var lo = -half;
    var hi = half;
    if (facing > 0.5) {
        lo = 0.0;
        hi = thickness;
    }
    if (facing < -0.5) {
        lo = -thickness;
        hi = 0.0;
    }
    return abs(profile_mitred(chain, miters, q, t) - 0.5 * (lo + hi)) - half;
}

// The exact signed distance to an outline from a point near it, by the segments its bake's block
// lists (backlog item 115): `x` is whether it answered, `y` the distance.
//
// Walks the block's list for the nearest segment, and answers only when that is within
// `LIST_REACH` margins, which is when no unlisted segment can be nearer. The sign is the nearest
// feature's: a point beside an edge is inside on the edge's inner side, and a point nearest a
// vertex is inside where the sum of its two edges' outward normals points away from -- the
// angle-weighted pseudonormal, exact for a polygon -- so no parity walk over the whole outline is
// needed. Nothing is answered off the grid, in a block that stores nothing, or in one with no
// segments listed, and the caller reads the grid there.
fn outline_distance_local(span: CurveSpan, q: vec2<f32>) -> vec2<f32> {
    let none = vec2<f32>(0.0, 0.0);
    let x = (q.x - span.lo_x) / span.cell;
    let y = (q.y - span.lo_y) / span.cell;
    if (x < 0.0 || y < 0.0 || x >= f32(span.bake_wide - 1u) || y >= f32(span.bake_high - 1u)) {
        return none;
    }
    let bi = min(u32(floor(x / 8.0)), span.blocks_wide - 1u);
    let bj = min(u32(floor(y / 8.0)), span.blocks_high - 1u);
    let k = bj * span.blocks_wide + bi;
    let first = u32(profile_bakes[span.segs_start + 2u * k]);
    let count = u32(profile_bakes[span.segs_start + 2u * k + 1u]);
    if (count == 0u) {
        return none;
    }
    var best = 1e30;
    var best_i = 0u;
    var best_t = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let i = u32(profile_bakes[span.seglist_start + first + j]);
        let a = curve_points[span.start + i].xy;
        let b = curve_points[span.start + select(i + 1u, 0u, i + 1u == span.count)].xy;
        let ab = b - a;
        let len2 = dot(ab, ab);
        if (len2 <= 0.0) {
            continue;
        }
        let t = clamp(dot(q - a, ab) / len2, 0.0, 1.0);
        let d = length(q - (a + ab * t));
        if (d < best) {
            best = d;
            best_i = i;
            best_t = t;
        }
    }
    if (best > LIST_REACH * span.margin) {
        return none;
    }
    let ccw = span.ccw > 0u;
    let a = curve_points[span.start + best_i].xy;
    let b = curve_points[span.start + select(best_i + 1u, 0u, best_i + 1u == span.count)].xy;
    var inside = false;
    if (best_t > 1e-5 && best_t < 1.0 - 1e-5) {
        let left = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x) > 0.0;
        inside = left == ccw;
    } else {
        // At a vertex: its two edges' outward normals, summed.
        var v = a;
        var e_in = vec2<f32>(0.0, 0.0);
        var e_out = vec2<f32>(0.0, 0.0);
        if (best_t <= 1e-5) {
            let prev = curve_points[span.start + select(best_i - 1u, span.count - 1u, best_i == 0u)].xy;
            e_in = a - prev;
            e_out = b - a;
        } else {
            v = b;
            let next_i = select(best_i + 1u, 0u, best_i + 1u == span.count);
            let next = curve_points[span.start + select(next_i + 1u, 0u, next_i + 1u == span.count)].xy;
            e_in = b - a;
            e_out = next - b;
        }
        let out_in = normalize(select(vec2<f32>(-e_in.y, e_in.x), vec2<f32>(e_in.y, -e_in.x), ccw));
        let out_out = normalize(select(vec2<f32>(-e_out.y, e_out.x), vec2<f32>(e_out.y, -e_out.x), ccw));
        inside = dot(q - v, out_in + out_out) < 0.0;
    }
    return vec2<f32>(1.0, select(best, -best, inside));
}

// One sample of a baked profile, bilinear, less the margin that keeps it a lower bound.
//
// Off the grid the answer is the hypotenuse of how far the query is from the grid and what is
// stored at the nearest point on it — a bound rather than a guess, because the profile lies
// inside the grid's box and the clamp is componentwise, so every axis contributes a longer leg.
// The same reasoning and the same arithmetic as the CPU's, which is what lets the two agree.
fn baked_profile_distance(span: CurveSpan, q: vec2<f32>) -> f32 {
    let last = f32(span.bake_wide - 1u);
    let tall = f32(span.bake_high - 1u);
    let x = (q.x - span.lo_x) / span.cell;
    let y = (q.y - span.lo_y) / span.cell;

    let cx = clamp(x, 0.0, last);
    let cy = clamp(y, 0.0, tall);
    let edge = sample_bake(span, cx, cy);
    if (x < 0.0 || y < 0.0 || x > last || y > tall) {
        let away = length(vec2<f32>((x - cx) * span.cell, (y - cy) * span.cell));
        return length(vec2<f32>(away, max(edge, 0.0)));
    }
    return edge;
}

// Bilinear at a position already on the grid: from the block if there is one, else coarse.
//
// The page says which. A block that is stored covers its own eight cells plus a one-sample
// apron, so interpolating inside it never reaches into a neighbour — which is what keeps this a
// lookup rather than a hunt for the next block. Where there is no block there is no surface
// either, and the coarse reading is enough with the wider margin a wider cell owes.
fn sample_bake(span: CurveSpan, x: f32, y: f32) -> f32 {
    let block = 8.0;
    let bi = min(u32(floor(x / block)), span.blocks_wide - 1u);
    let bj = min(u32(floor(y / block)), span.blocks_high - 1u);
    let page = profile_bakes[span.page_start + bj * span.blocks_wide + bi];

    // Negative is "no block": an index cannot be, and it keeps the page in the same buffer as
    // everything else rather than costing a binding of its own.
    if (page >= 0.0) {
        let start = span.fine_start + u32(page);
        let lx = x - f32(bi) * block;
        let ly = y - f32(bj) * block;
        let i = min(u32(floor(lx)), 7u);
        let j = min(u32(floor(ly)), 7u);
        let stride = 9u;
        let base = start + j * stride + i;
        let fx = lx - f32(i);
        let fy = ly - f32(j);
        let top = mix(profile_bakes[base], profile_bakes[base + 1u], fx);
        let bottom = mix(profile_bakes[base + stride], profile_bakes[base + stride + 1u], fx);
        return mix(top, bottom, fy) - span.margin;
    }

    let gx = x / block;
    let gy = y / block;
    let ci = min(u32(floor(gx)), span.blocks_wide - 1u);
    let cj = min(u32(floor(gy)), span.blocks_high - 1u);
    let stride = span.blocks_wide + 1u;
    let base = span.coarse_start + cj * stride + ci;
    let fx = gx - f32(ci);
    let fy = gy - f32(cj);
    let top = mix(profile_bakes[base], profile_bakes[base + 1u], fx);
    let bottom = mix(profile_bakes[base + stride], profile_bakes[base + stride + 1u], fx);
    // The coarse margin toward nought, off whichever sign the reading has: a bound on the
    // distance outside and on the depth inside. Subtracted from a negative reading it made a
    // section deeper than it is, and a slot cut with one is marched by that depth. The CPU
    // does the same, in `BakedProfile::sample`, and `baked_profile.rs` says why the fine level
    // above is not treated this way.
    let here = mix(top, bottom, fy);
    return here - sign(here) * span.coarse_margin;
}

// A profile given thickness along its own normal, centred on its plane.
//
// Outside both the profile and the slab the true distance is the diagonal, not the larger of
// the two — taking the max there understates it, which is safe but makes a ray crawl along the
// rim. Matches the CPU exactly.
// ---- profile solids: instantiated again for the normal pass ----
//
// **Every `fn` from here to the end marker is emitted a second time by codegen, renamed `_bake`,
// with each call to another of them renamed too and `profile_distance` read as
// `profile_distance_bake`.** The normal pass calls the copies. The reason is the driver's
// compile: the value pass must finish on the outline (item 115) and walk a section without a
// bake, and both are loops, inlined wherever a section is read -- and a differenced normal reads
// a section six or seven times per solid, inside a loop of its own. Four partial revolves took
// 9.7 s to compile with the normal pass reading the exact section and 2.5 s with it reading the
// bake alone; four extrudes 8.2 s (`document_cost`, 2026-10-19, after the wedge's own loops had
// taken it from 106 s); four sweeps 27.6 s with the normal pass and 5.2 s without it, four lofts
// 6.5 s and 3.7 s, which is why the region runs through the sweeps and the lofts to the pipe.
// A normal from the bake is the bilinear grid's gradient, which the normal was before item 115
// and is accurate to the cell; the surface itself is the value pass's and unchanged. Structs
// and constants in the region are not copied; `profile_mitred` is not a section with a bake and
// walks in both copies, so a drafted extrude with mitred corners keeps its cost.
// `the_normal_pass_reads_every_profile_solid_through_its_bake_copy` holds the list.

// A profile swept along `dir`, given in the profile's own frame.
//
// One function for the straight sweep and the leaning one, because the straight sweep *is* the
// leaning one with `dir = (0, 0, 1)`: the slide back to the plane becomes a straight drop, the
// correction factor becomes 1, and every line below reduces to what the straight version used to
// say. Two functions would have been two things to keep agreeing.
// `symmetric` is 1.0 when the depth is measured about the plane rather than from it: half each
// way, with the profile as the widest section and the draft opening away from it on both sides.
// A number rather than a generated branch so that ticking the box refills a buffer instead of
// recompiling the shader.
// How an extrusion's depth lies about its plane, as the six slots the value pass pushes after
// the nine of its placement, in that order: side one's depth, the tangent of its draft, which
// way the depth lies (0 one side, 1 half each way, 2 two sides), where along the sweep the
// material begins, and side two's depth and draft tangent, read only for two sides. One struct
// rather than six scalars, so that a function taking an extrude's numbers takes them once and
// the run loops build it from their stride.
struct ExtrudeDepth {
    height: f32,
    draft: f32,
    sides: f32,
    start: f32,
    height_2: f32,
    draft_2: f32,
}

fn extrude_distance(index: u32, p: vec3<f32>, dir: vec3<f32>, depth: ExtrudeDepth) -> f32 {
    return extrude_rim_distance(index, p, dir, depth, NO_RIM);
}

// **A hard rim, which is what every extrude without one passes.** The rimmed functions below
// are the bodies and the plain names call them with this, so an extrude that has not asked
// compiles to the call it always did and `extrude_combine` decides the rest at a constant the
// compiler folds.
const NO_RIM: vec4<f32> = vec4<f32>(0.0, 2.0, 0.0, 0.0);

// `rim` is the reach, the shape's exponent, which ends -- 0 both, 1 the start, 2 the end -- and
// the square-corner scale, nought for the max-norm bound: `fieldform_field::Rim`, and `Rim::cut`
// for which solids read the exact corner and why the others keep the bound.
fn extrude_rim_distance(
    index: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    rim: vec4<f32>,
) -> f32 {
    let along = abs(dir.z);
    if (along < 1e-6) {
        return 1e30;
    }
    let t = p.z / dir.z;
    let landed = p - dir * t;
    return extrude_over_rim(profile_distance(index, landed.xy), t, dir, depth, rim);
}

// An extrusion whose draft **mitres** the section's corners rather than rounding them.
//
// The same solid as `extrude_distance` in every other respect -- the same landing, the same
// slab, the same corner and the same corrections -- and the one difference is that the section
// is asked for its own offset instead of being handed a number to subtract. A rounded draft
// keeps the call it always had, byte for byte, so a document that has not asked compiles to
// what it compiled to.
fn extrude_sharp_distance(
    index: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
) -> f32 {
    return extrude_sharp_rim_distance(index, miters, p, dir, depth, NO_RIM);
}

fn extrude_sharp_rim_distance(
    index: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    rim: vec4<f32>,
) -> f32 {
    if (abs(dir.z) < 1e-6) {
        return 1e30;
    }
    let t = p.z / dir.z;
    let landed = p - dir * t;
    let grown = extrude_grown(t, depth);
    return extrude_combine_rim(
        profile_mitred(index, miters, landed.xy, grown.x),
        grown.y,
        grown.w,
        dir,
        grown.z,
        rim
    );
}

// The same for a wall round a closed loop: the loop's mitred reading, folded about the band.
fn wall_extrude_sharp_distance(
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    wall: vec4<f32>,
) -> f32 {
    return wall_extrude_sharp_rim_distance(chain, miters, p, dir, depth, wall, NO_RIM);
}

fn wall_extrude_sharp_rim_distance(
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    wall: vec4<f32>,
    rim: vec4<f32>,
) -> f32 {
    if (abs(dir.z) < 1e-6) {
        return 1e30;
    }
    let t = p.z / dir.z;
    let landed = p - dir * t;
    let grown = extrude_grown(t, depth);
    return extrude_combine_rim(
        wall_mitred(chain, miters, landed.xy, grown.x, wall.x, wall.y),
        grown.y,
        grown.w,
        dir,
        grown.z,
        rim
    );
}

// What an extrusion makes of a reading taken in its plane, `t` along the sweep from it.
//
// **Apart from `extrude_distance` so that a wall goes through the very same lines.** A thin
// extrude differs from a thick one in what is read in the plane and in nothing else, so the
// slab, the draft, the corner and the two corrections are written once and both call them. A
// copy would have been a second thing to keep agreeing with `Extrusion`.
fn extrude_over(reading: f32, t_from_plane: f32, dir: vec3<f32>, depth: ExtrudeDepth) -> f32 {
    return extrude_over_rim(reading, t_from_plane, dir, depth, NO_RIM);
}

fn extrude_over_rim(
    reading: f32,
    t_from_plane: f32,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    rim: vec4<f32>,
) -> f32 {
    let grown = extrude_grown(t_from_plane, depth);
    return extrude_combine_rim(reading - grown.x, grown.y, grown.w, dir, grown.z, rim);
}

// **How far the draft has moved the wall, the distance out of the slab, the steeper of the two
// drafts, and how far past the middle of the length the point is, positive toward the end** --
// what `extrude_over` needs before it has a reading, and what a mitred section needs *instead
// of* a reading to subtract from. `grown_and_slab` and `toward_end` are the kernel's same split;
// the last is what lets a rim on one end tell which end is nearer.
fn extrude_grown(t_from_plane: f32, depth: ExtrudeDepth) -> vec4<f32> {
    // From the start plane, which is where the material begins and so the base a draft is
    // measured from; the section is still read in the profile's own plane.
    let t = t_from_plane - depth.start;
    // **A draft is an offset, not a scale.** Subtracting a constant from a distance gives the
    // distance to the offset shape, so every wall moves out by the same amount along its own
    // normal -- which is what a draft angle means. A scale would drag a profile that is not
    // centred on its own origin sideways as it tapered. Matches `plane_and_slab` exactly, arm
    // for arm: one side from the start, half each way about it with the profile the widest
    // section, or two sides each growing from the start by its own draft.
    var grown: f32;
    var slab: f32;
    var toward: f32;
    var steepest = abs(depth.draft);
    if (depth.sides > 1.5) {
        let forward = select(1.0, -1.0, depth.height < 0.0);
        let u = t * forward;
        let one = abs(depth.height);
        let two = abs(depth.height_2);
        grown = select(-u * depth.draft_2, u * depth.draft, u >= 0.0);
        toward = u - (one - two) * 0.5;
        slab = abs(toward) - (one + two) * 0.5;
        steepest = max(steepest, abs(depth.draft_2));
    } else if (depth.sides > 0.5) {
        grown = abs(t) * depth.draft;
        toward = t;
        slab = abs(t) - abs(depth.height) * 0.5;
    } else {
        // Zero at the start and `height` at the far end. Either sign, since a negative depth is
        // how one is flipped, and measured along the extrusion so it still opens away.
        let half = depth.height * 0.5;
        grown = t * sign(depth.height) * depth.draft;
        toward = (t - half) * sign(depth.height);
        slab = abs(t - half) - abs(half);
    }
    return vec4<f32>(grown, slab, steepest, toward);
}

// The section's reading and the slab's, made the extrusion's: the corner where the two are
// square, and the correction a lean and a draft together cost.
fn extrude_combine(plane: f32, slab: f32, dir: vec3<f32>, steepest: f32) -> f32 {
    return extrude_combine_rim(plane, slab, 0.0, dir, steepest, NO_RIM);
}

// A rim's join, matching `Rim::join`: the blend's bound with both signs turned, at the shape's
// exponent. Moves no faster than the faster of its two readings at any angle between them.
// A rim's shape and its second leg, from the vector's `y`: the exponent, or -- negative -- a
// chamfer whose leg along the end is `−y` (see `rim_vector` in codegen). `x` the exponent, `y`
// the second leg or nought.
fn rim_shape(rim: vec4<f32>) -> vec2<f32> {
    return select(vec2<f32>(rim.y, 0.0), vec2<f32>(1.0, -rim.y), rim.y < 0.0);
}

// The readings scaled so a bevel of two legs is the equal-legged bevel's zero set, and the
// number it is read at -- `Rim::scaled`: `x·m/s_y`, `y·m/s_x`, `m = min(s_x, s_y)`, the scales
// at most one so every bound holds as it did. `(x, y, k)` unchanged with one leg.
fn rim_two_legs(x: f32, y: f32, rim: vec4<f32>) -> vec3<f32> {
    let legs = rim_shape(rim);
    if (legs.y > 0.0 && rim.x > 0.0) {
        let m = min(rim.x, legs.y);
        return vec3<f32>(x * m / legs.y, y * m / rim.x, m);
    }
    return vec3<f32>(x, y, rim.x);
}

fn rim_join(x: f32, y: f32, rim: vec4<f32>) -> f32 {
    if (rim.w > 0.0) {
        return rim_join_square(x, y, rim);
    }
    // **What this bound reads short by along the rim's normal, for the still's occlusion** --
    // the blend's arrangement (`seam_scale`), with the angle known rather than measured: a
    // rim's two readings are a section's and a slab's, square but for a lean, and the max-norm
    // bound reads `cos 45°` of a step along the arc's normal (spec `round-chamfer-spec.md` §4.4:
    // 0.707 on every bound rim). Left only where the arc is the nearest piece, which is where
    // both readings are within the reach and no further apart than it; a rim's normal is
    // differenced, so this is the one place that knows.
    let k = rim.x;
    let on_arc = k > 0.0 && x > -k && y > -k && abs(x - y) <= k;
    seam_scale = select(1.0, 1.4142135, on_arc);
    let s = rim_two_legs(x, y, rim);
    return -blend_bound(-s.x, -s.y, s.z, rim_shape(rim).x);
}

// The exact rounded corner of the (x, y) plane -- `edged2`, the two-axis form the cylinder's
// edges read -- through the scale the readings' lean can cost, and never under the hard corner;
// `y` clamped at the reach and the rest taken back, as `Rim::join_square` says why.
fn rim_join_square(x: f32, y: f32, rim: vec4<f32>) -> f32 {
    let k = rim.x;
    let yy = select(y, min(y, k), rim.w > 1.0);
    let shape = rim_shape(rim);
    var near: f32;
    if (shape.y > 0.0) {
        // A bevel of two legs between square faces, exactly: the hard corner, or the bevel's
        // plane `s_x·x + s_y·y + s_x·s_y = 0` read as the true distance to it (`Rim::join_square`).
        let hard = edged2(vec2<f32>(x, yy), vec2<f32>(0.0, 1.0));
        let bevel = (k * x + shape.y * yy + k * shape.y) / length(vec2<f32>(k, shape.y));
        near = max(hard, bevel) - (y - yy);
    } else {
        near = edged2(vec2<f32>(x, yy), vec2<f32>(k, shape.x)) - (y - yy);
    }
    // Exact at the circle and the chamfer; the cube norm of the G2 corner reads 0.891 of a step
    // along its normal (measured in `rim_ao_reading`), put back as `rim_join` puts back its own.
    let g2_corner = shape.x != 1.0 && shape.x != 2.0 && k > 0.0 && x > -k && y > -k;
    seam_scale = select(1.0, 1.122, g2_corner);
    return max(near / rim.w, max(x, y));
}

// The same where only one end may be rounded -- `Rim::join_ends`: `slab` the reading along the
// length, `toward` how far past its middle the point sits, positive toward the end. The rounded
// end's own slab is joined and the other end's taken hard, which at both ends is `rim_join`.
fn rim_join_ends(x: f32, slab: f32, toward: f32, rim: vec4<f32>) -> f32 {
    if (rim.z < 0.5) {
        return rim_join(x, slab, rim);
    }
    let half = abs(toward) - slab;
    let rounded = select(-toward, toward, rim.z > 1.5);
    return max(rim_join(x, rounded - half, rim), -rounded - half);
}

fn extrude_combine_rim(
    plane: f32,
    slab: f32,
    toward: f32,
    dir: vec3<f32>,
    steepest: f32,
    rim: vec4<f32>,
) -> f32 {
    let along = abs(dir.z);
    let lean_of = sqrt(max(1.0 - along * along, 0.0));
    if (rim.x > 0.0) {
        // Through the blend's bound, so the lean's and the draft's divisor is all it needs.
        return rim_join_ends(plane, slab, toward, rim) * along
            * inverseSqrt(along * along + (lean_of + steepest) * (lean_of + steepest));
    }
    // The corner is Pythagoras between two *perpendicular* measurements. A lean tilts one and a
    // draft tilts the other, so it is taken only when there is neither.
    if (along >= 1.0 && steepest == 0.0 && plane > 0.0 && slab > 0.0) {
        return length(vec2<f32>(plane, slab));
    }
    // **A lean and a draft together cost more than each alone.** The reading's rate along the
    // normal is `(sin(lean) + tan(draft)) / cos(lean)`, since the shear moves where the section
    // is read and the draft takes from the reading for the same move, and across the plane it
    // is one. So the bound is `sqrt(cos^2 + (sin + tan)^2) / cos`: `1 / cos` with no draft,
    // `sqrt(1 + tan^2)` with no lean, and more than their product with both -- which is what
    // this used to charge, and which claimed 2.4 mm of room that was not there on a 25-degree
    // draft along a lean (`SkewExtrusion::distance`, `examples/lean_draft_probe.rs`). With two
    // sides the slope is the steeper side's.
    let lean = sqrt(max(1.0 - along * along, 0.0));
    return max(plane, slab) * along * inverseSqrt(along * along + (lean + steepest) * (lean + steepest));
}

// A profile swept about the frame's own axis.
//
// Exact rather than a bound: the nearest point on a surface of revolution lies in the query's
// own half-plane, so the distance in (r, z) is the distance in space.
fn revolve_distance(index: u32, p: vec3<f32>) -> f32 {
    return profile_distance(index, vec2<f32>(length(p.xy), p.z));
}

// Outward normal of an extrusion, in world coordinates.
//
// Central-differenced rather than derived, and deliberately so: the CPU has no analytic normal
// for these either — a profile is a polyline, and its gradient is piecewise constant with a
// discontinuity at every vertex — so differencing is what both sides do and the two agree.
//
// The point arrives in the block's own frame, so the gradient comes out in that frame and has
// to be turned back. `euler_rotation` is the same matrix `to_local` used, applied the other way
// round.
// The same for a draft that mitres its corners: the mitred reading differenced the same way.
// Six readings rather than an analytic normal, for the reason `extrude_normal` gives -- a
// section is a polyline and its gradient is piecewise constant, so there is nothing to derive.
fn extrude_sharp_normal(
    index: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    euler: vec3<f32>,
) -> vec3<f32> {
    return extrude_sharp_rim_normal(index, miters, p, dir, depth, euler, NO_RIM);
}

fn extrude_sharp_rim_normal(
    index: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    euler: vec3<f32>,
    rim: vec4<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    let n = vec3<f32>(
        extrude_sharp_rim_distance(index, miters, p + vec3<f32>(h, 0.0, 0.0), dir, depth, rim)
            - extrude_sharp_rim_distance(index, miters, p - vec3<f32>(h, 0.0, 0.0), dir, depth, rim),
        extrude_sharp_rim_distance(index, miters, p + vec3<f32>(0.0, h, 0.0), dir, depth, rim)
            - extrude_sharp_rim_distance(index, miters, p - vec3<f32>(0.0, h, 0.0), dir, depth, rim),
        extrude_sharp_rim_distance(index, miters, p + vec3<f32>(0.0, 0.0, h), dir, depth, rim)
            - extrude_sharp_rim_distance(index, miters, p - vec3<f32>(0.0, 0.0, h), dir, depth, rim),
    );
    return normalize(euler_rotation(euler) * n);
}

// And for a wall round a loop under a mitred draft.
fn wall_extrude_sharp_normal(
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    wall: vec4<f32>,
    euler: vec3<f32>,
) -> vec3<f32> {
    return wall_extrude_sharp_rim_normal(chain, miters, p, dir, depth, wall, euler, NO_RIM);
}

fn wall_extrude_sharp_rim_normal(
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    wall: vec4<f32>,
    euler: vec3<f32>,
    rim: vec4<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    let n = vec3<f32>(
        wall_extrude_sharp_rim_distance(chain, miters, p + vec3<f32>(h, 0.0, 0.0), dir, depth, wall, rim)
            - wall_extrude_sharp_rim_distance(chain, miters, p - vec3<f32>(h, 0.0, 0.0), dir, depth, wall, rim),
        wall_extrude_sharp_rim_distance(chain, miters, p + vec3<f32>(0.0, h, 0.0), dir, depth, wall, rim)
            - wall_extrude_sharp_rim_distance(chain, miters, p - vec3<f32>(0.0, h, 0.0), dir, depth, wall, rim),
        wall_extrude_sharp_rim_distance(chain, miters, p + vec3<f32>(0.0, 0.0, h), dir, depth, wall, rim)
            - wall_extrude_sharp_rim_distance(chain, miters, p - vec3<f32>(0.0, 0.0, h), dir, depth, wall, rim),
    );
    return normalize(euler_rotation(euler) * n);
}

fn extrude_normal(
    index: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    euler: vec3<f32>,
) -> vec3<f32> {
    return extrude_rim_normal(index, p, dir, depth, euler, NO_RIM);
}

// In one loop and not six calls, as the wedge's is and for its reason.
fn extrude_rim_normal(
    index: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    euler: vec3<f32>,
    rim: vec4<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * extrude_rim_distance(index, p + step, dir, depth, rim);
    }
    return normalize(euler_rotation(euler) * n);
}

// A wall turned about the frame's own axis: a thin revolve. `wall` is its four numbers, in the
// order `wall_distance` takes them.
//
// **Read on both sides of the axis where the band crosses it**, which `mirrored` says: the
// curve is laid in the half-plane by its points, and its band then goes a thickness to one
// side, which near the axis is across it. That part turns out the radii it would have had on
// this side, so the reading is the nearer of the wall at `(r, z)` and at `(-r, z)`.
// `Footprint::about_axis` says why that is exact outside the material and does the same.
//
// A loop whose bound is a number rather than two calls, so the wall is inlined once: the
// second reading is taken only where there is a band across the axis to take it of.
fn wall_revolve_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
) -> f32 {
    let r = length(p.xy);
    let passes = 1u + u32(mirrored > 0.5);
    var nearest = 1e30;
    for (var k = 0u; k < passes; k = k + 1u) {
        let side = 1.0 - 2.0 * f32(k);
        nearest = min(
            nearest,
            wall_distance(chain, faces, sides, vec2<f32>(side * r, p.z), wall.x, wall.y, wall.z, wall.w)
        );
    }
    return nearest;
}

// Its outward normal, in world coordinates: differenced, for the reason `extrude_normal` is,
// and in one loop for the reason `wall_sweep_value_normal` is.
fn wall_revolve_normal(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
    euler: vec3<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * wall_revolve_distance(chain, faces, sides, p + step, wall, mirrored);
    }
    return normalize(euler_rotation(euler) * n);
}

// In one loop and not six calls, as the wedge's is and for its reason.
fn revolve_normal(index: u32, p: vec3<f32>, euler: vec3<f32>) -> vec3<f32> {
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * revolve_distance(index, p + step);
    }
    return normalize(euler_rotation(euler) * n);
}

// A section swept part of the way about the axis: the wedge of a revolve between two
// half-planes, `from` radians round from +x and `sweep` radians wide. The same three cases
// `RevolvedWedge::distance` takes, in the same order, and it has to stay the same:
//
// - Inside the wedge and outside the solid, the half-plane's own reading, exactly.
// - Outside the wedge, the nearer of the two caps -- each the section as a planar region in its
//   own half-plane, read with `u` signed across the axis, Pythagoras against the height off its
//   plane. The angle to each cap is taken the smaller way round.
// - Inside the wedge and inside the solid, the depth bounded by each cap's plane, or by the axis
//   past ninety degrees. A bound, which is why the leaf says `LowerBound`.
//
// `cap` and `in_wedge` are what the profile's and the wall's share; what differs is what is
// read in the plane, which is why the two kinds are two functions over these.
fn wedge_angles(p: vec3<f32>, begins: f32, sweep: f32) -> vec2<f32> {
    let tau = 6.283185307179586;
    var rel = atan2(p.y, p.x) - begins;
    rel = rel - tau * floor(rel / tau);
    return vec2<f32>(rel, length(p.xy));
}

fn cap_distance(d: f32, h: f32) -> f32 {
    return select(abs(h), length(vec2<f32>(d, h)), d > 0.0);
}

// The depth inside the wedge, bounded by its two caps: `d` is the half-plane's own reading.
fn wedge_depth(d: f32, rel: f32, sweep: f32, r: f32) -> f32 {
    let quarter = 1.5707963267948966;
    let to_start = -(r * sin(min(rel, quarter)));
    let to_end = -(r * sin(min(sweep - rel, quarter)));
    return max(max(d, to_start), to_end);
}

// Where a point outside the wedge lands in each cap's plane: `u` across the axis and `h` off
// the plane, for the start cap in `xy` and the end cap in `zw`.
fn wedge_landings(rel: f32, sweep: f32, r: f32) -> vec4<f32> {
    let tau = 6.283185307179586;
    let before = tau - rel;
    let past = rel - sweep;
    let psi_a = min(before, tau - before);
    let psi_b = min(past, tau - past);
    return vec4<f32>(r * cos(psi_a), r * sin(psi_a), r * cos(psi_b), r * sin(psi_b));
}

fn revolve_wedge_distance(index: u32, p: vec3<f32>, begins: f32, sweep: f32) -> f32 {
    return revolve_wedge_rim_distance(index, p, begins, sweep, NO_RIM);
}

fn revolve_wedge_normal(index: u32, p: vec3<f32>, begins: f32, sweep: f32, euler: vec3<f32>) -> vec3<f32> {
    return revolve_wedge_rim_normal(index, p, begins, sweep, NO_RIM, euler);
}

// Whether cap `which` -- 1 the start, 2 the end -- is rounded by `rim`, whose `z` is 0 for both
// ends, 1 the start, 2 the end.
fn wedge_cap_rounded(rim: vec4<f32>, which: f32) -> bool {
    return rim.x > 0.0 && (rim.z < 0.5 || abs(rim.z - which) < 0.5);
}

// The section's reading against a cap's, inside the wedge: the blend's bound where the cap is
// rounded, the plain intersection where it is hard -- `RevolvedWedge::against`. With both caps
// hard this is `max(d, to_start, to_end)`, which is `wedge_depth` where `d < 0` and `d` where it
// is not: a hard wedge reads what it always did.
fn wedge_against(x: f32, y: f32, rim: vec4<f32>, which: f32) -> f32 {
    return select(max(x, y), rim_join(x, y, rim), wedge_cap_rounded(rim, which));
}

// A rounded cap seen from outside the wedge: the same line, continued across the cap plane
// with the sign of `y` turned, at the query's own radius -- `RevolvedWedge::distance` says why
// the landing read a hard cap takes would not be a bound here.
fn wedge_rim_cap(x: f32, r: f32, psi: f32, rim: vec4<f32>) -> f32 {
    let quarter = 1.5707963267948966;
    return rim_join(x, r * sin(min(psi, quarter)), rim);
}

// **The section is read in one loop, up to three times, and not at three places**, so the
// outline walk it is (`profile_distance`) is inlined once per wedge reading rather than once
// per place: the query's own meridian always, and where each hard cap lands when the query is
// outside the wedge. The count is what the query decides, so the driver has no literal to
// unroll. Measured with `revolve_wedge_rim_normal`'s loop already in: 19.6 s to 7-9 s for four
// partial revolves, cold (`document_cost` on `hard.fform`, 2026-10-18), and one wedge 4.0 s to
// 0.95 s -- a bare sphere is 0.7 s.
fn revolve_wedge_rim_distance(index: u32, p: vec3<f32>, begins: f32, sweep: f32, rim: vec4<f32>) -> f32 {
    let tau = 6.283185307179586;
    let quarter = 1.5707963267948966;
    let a = wedge_angles(p, begins, sweep);
    let rel = a.x;
    let r = a.y;
    let outside = rel > sweep;
    // The angle to each cap, the smaller way round. Meaningless inside the wedge, and not read
    // there: `reads` is one.
    let before = tau - rel;
    let past = rel - sweep;
    let psi = vec2<f32>(min(before, tau - before), min(past, tau - past));
    let u = vec3<f32>(r, r * cos(psi.x), r * cos(psi.y));
    var d = vec3<f32>(0.0, 0.0, 0.0);
    let reads = select(1u, 3u, outside);
    for (var k = 0u; k < reads; k = k + 1u) {
        // A rounded cap reads the query's own meridian (`wedge_rim_cap`), never its landing.
        if (k == 0u || !wedge_cap_rounded(rim, f32(k))) {
            d[k] = profile_distance(index, vec2<f32>(u[k], p.z));
        }
    }
    let x = d.x;
    if (!outside) {
        let to_start = -(r * sin(min(rel, quarter)));
        let to_end = -(r * sin(min(sweep - rel, quarter)));
        return max(wedge_against(x, to_start, rim, 1.0), wedge_against(x, to_end, rim, 2.0));
    }
    var at_start: f32;
    if (wedge_cap_rounded(rim, 1.0)) {
        at_start = wedge_rim_cap(x, r, psi.x, rim);
    } else {
        at_start = cap_distance(d.y, r * sin(psi.x));
    }
    var at_end: f32;
    if (wedge_cap_rounded(rim, 2.0)) {
        at_end = wedge_rim_cap(x, r, psi.y, rim);
    } else {
        at_end = cap_distance(d.z, r * sin(psi.y));
    }
    return min(at_start, at_end);
}

// **Differenced in one loop and not six calls**, for the reason `wall_sweep_value_normal` gives,
// and measured here because a wedge is the heaviest profile reading there is: outside the wedge
// a hard cap reads the section where it lands as well as where the query is, so one wedge
// reading is three outline walks (`profile_distance`), and six calls written out are eighteen
// walks inlined into one function, per wedge. Four partial revolves took the driver 106 s to
// compile the march that way (`document_cost` on `hard.fform`, 2026-10-18); with the normal cut
// out altogether it took 9 s, and with the two landing reads cut out of each copy 12 s, so the
// copies were the whole of it. As one loop, 19.6 s; the literal six is not what the driver
// unrolls -- an opaque count measured the same -- and `revolve_wedge_rim_distance`'s own loop
// takes the rest. Spec `round-chamfer-spec.md` §4.4 keeps the table.
fn revolve_wedge_rim_normal(index: u32, p: vec3<f32>, begins: f32, sweep: f32, rim: vec4<f32>, euler: vec3<f32>) -> vec3<f32> {
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * revolve_wedge_rim_distance(index, p + step, begins, sweep, rim);
    }
    return normalize(euler_rotation(euler) * n);
}

// The wall's reading in the whole plane through the axis, `u` signed across it, mirrored where
// the band crosses the axis -- `Footprint::across_axis`, for a cap.
// One loop and not two calls, as `wall_revolve_distance` is, and for its reason: a wall is the
// heaviest thing the prelude reads, and a second call is a second copy of it.
fn wall_across_axis(chain: u32, faces: u32, sides: u32, u: f32, z: f32, wall: vec4<f32>, mirrored: f32) -> f32 {
    let passes = 1u + u32(mirrored > 0.5);
    var nearest = 1e30;
    for (var k = 0u; k < passes; k = k + 1u) {
        let side = 1.0 - 2.0 * f32(k);
        nearest = min(
            nearest,
            wall_distance(chain, faces, sides, vec2<f32>(side * u, z), wall.x, wall.y, wall.z, wall.w)
        );
    }
    return nearest;
}

fn wall_revolve_wedge_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
    begins: f32,
    sweep: f32,
) -> f32 {
    return wall_revolve_wedge_rim_distance(chain, faces, sides, p, wall, mirrored, begins, sweep, NO_RIM);
}

// The profile's reading above, over a wall: the own-radius read is `wall_revolve_distance`, the
// landing read `wall_across_axis`.
fn wall_revolve_wedge_rim_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
    begins: f32,
    sweep: f32,
    rim: vec4<f32>,
) -> f32 {
    let tau = 6.283185307179586;
    let quarter = 1.5707963267948966;
    let a = wedge_angles(p, begins, sweep);
    let rel = a.x;
    let r = a.y;
    let outside = rel > sweep;
    let before = tau - rel;
    let past = rel - sweep;
    let psi = vec2<f32>(min(before, tau - before), min(past, tau - past));
    // The wall read in one loop at up to three places, as `revolve_wedge_rim_distance` reads the
    // section, and for the same reason. The query's own reading is the wall across the axis at
    // its own radius: `wall_revolve_distance` is `wall_across_axis` at `u = r`.
    let u = vec3<f32>(r, r * cos(psi.x), r * cos(psi.y));
    var d = vec3<f32>(0.0, 0.0, 0.0);
    let reads = select(1u, 3u, outside);
    for (var k = 0u; k < reads; k = k + 1u) {
        if (k == 0u || !wedge_cap_rounded(rim, f32(k))) {
            d[k] = wall_across_axis(chain, faces, sides, u[k], p.z, wall, mirrored);
        }
    }
    let x = d.x;
    if (!outside) {
        let to_start = -(r * sin(min(rel, quarter)));
        let to_end = -(r * sin(min(sweep - rel, quarter)));
        return max(wedge_against(x, to_start, rim, 1.0), wedge_against(x, to_end, rim, 2.0));
    }
    var at_start: f32;
    if (wedge_cap_rounded(rim, 1.0)) {
        at_start = wedge_rim_cap(x, r, psi.x, rim);
    } else {
        at_start = cap_distance(d.y, r * sin(psi.x));
    }
    var at_end: f32;
    if (wedge_cap_rounded(rim, 2.0)) {
        at_end = wedge_rim_cap(x, r, psi.y, rim);
    } else {
        at_end = cap_distance(d.z, r * sin(psi.y));
    }
    return min(at_start, at_end);
}

// Differenced in one loop, for the reason `wall_revolve_normal` is.
fn wall_revolve_wedge_normal(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
    begins: f32,
    sweep: f32,
    euler: vec3<f32>,
) -> vec3<f32> {
    return wall_revolve_wedge_rim_normal(chain, faces, sides, p, wall, mirrored, begins, sweep, NO_RIM, euler);
}

fn wall_revolve_wedge_rim_normal(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    wall: vec4<f32>,
    mirrored: f32,
    begins: f32,
    sweep: f32,
    rim: vec4<f32>,
    euler: vec3<f32>,
) -> vec3<f32> {
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * wall_revolve_wedge_rim_distance(chain, faces, sides, p + step, wall, mirrored, begins, sweep, rim);
    }
    return normalize(euler_rotation(euler) * n);
}

// ---------------------------------------------------------------------------------------------
// A curve given a thickness: the wall a thin extrude is built on.
//
// The same arithmetic `Wall::distance` does on the other side, in the same order, and it has to
// stay the same. `fieldform-field/src/wall.rs` carries the reasoning; this carries the lines.
//
// Three spans are read. `chain` is the curve, with what was decided about the run starting at
// each point in that point's `z`: one if its band is mitred at its start, two if at its end,
// four if the joint at its start fans out to the left. Those are decided once, on the CPU, and
// sent -- a turn of exactly a right angle is a mitre or it is not, and deciding it again here
// in single precision could decide it the other way. `faces` and `sides` are the wall's bare
// flat edges as pairs of points, found when the wall was built: the first is read from outside
// the wall as well as in, the second only from inside.
//
// **And none of the three is read when the chain's span carries a bake.** The grid is of the
// wall's own distance -- the band's, not the curve's -- so it is the whole answer, and it is
// asked first: one lookup where the walk below reads every run, which on a curve of 256 chords
// was five times the frame of the same solid drawn as a profile (`wall_cost.rs`).
// ---------------------------------------------------------------------------------------------

// The z of `a x b`: positive when `b` is to the left of `a`.
fn wall_cross(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// A quarter turn to the left.
fn wall_left(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-v.y, v.x);
}

fn wall_to_segment(q: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(q - (a + ab * t));
}

// How near `q` is to the nearest of a span's stretches, each a pair of points. A span of none
// is a wall with no such edge, and answers that nothing is near.
fn wall_nearest_stretch(index: u32, q: vec2<f32>) -> f32 {
    let span = curve_spans[index];
    var best = 1e30;
    for (var i = 0u; i + 1u < span.count; i = i + 2u) {
        best = min(
            best,
            wall_to_segment(q, curve_points[span.start + i].xy, curve_points[span.start + i + 1u].xy)
        );
    }
    return best;
}

// Whether a mitre has cut the band short on the fibre `q` is on, this side of `q`.
//
// `on_curve` and `at_query` are how far the two ends of the fibre -- its foot on the curve, and
// `q` itself -- lie to the kept side of the cut. The crossing is the share of the way along at
// which that reaches nought, and it only matters if it falls within the band.
fn wall_cut_short(lateral: f32, on_curve: f32, at_query: f32, thickness: f32) -> bool {
    return at_query < 0.0 && abs(lateral) * on_curve / (on_curve - at_query) < thickness;
}

// One round end of a one-sided wall: `x` is the signed distance to its disc, and `y` how near
// its rim is as an edge of the wall. Behind the end the rim is buried in the end run's own band
// -- unless that run is too short to bury it, or mitred -- and is then no nearer than its tips.
fn wall_round_end(
    q: vec2<f32>,
    at: vec2<f32>,
    along: vec2<f32>,
    beyond: f32,
    middle: f32,
    half: f32,
    buried_behind: bool,
) -> vec2<f32> {
    let across = wall_left(along);
    let from_centre = q - (at + across * middle);
    let d = length(from_centre) - half;
    var rim = abs(d);
    if (dot(from_centre, along) * beyond < 0.0 && buried_behind) {
        rim = min(length(from_centre - across * half), length(from_centre + across * half));
    }
    return vec2<f32>(d, rim);
}

// The signed distance from `q` to a wall, in the wall's own plane: negative in its material.
//
// `facing` is which way the band lies from the curve: `+1` to the left of an open chain and
// outward from a loop, `-1` the other way, nought about the curve. `round` and `closed` are
// flags as numbers, so that changing either refills a buffer instead of rebuilding the shader.
fn wall_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    q: vec2<f32>,
    thickness: f32,
    facing: f32,
    round: f32,
    closed: f32,
) -> f32 {
    let half = 0.5 * thickness;
    var lo = -half;
    var hi = half;
    if (facing > 0.5) {
        lo = 0.0;
        hi = thickness;
    }
    if (facing < -0.5) {
        lo = -thickness;
        hi = 0.0;
    }
    let middle = 0.5 * (lo + hi);

    let span = curve_spans[chain];
    // A bake on this span is the wall's, never the loop's: `Wall::baked` samples the folded
    // distance, so there is nothing left to fold and the reading is returned as it is. This
    // has to come before the loop below, which would otherwise hand the same grid to
    // `profile_distance` and fold a wall's distance as though it were a curve's. A wall's bake
    // lists no segments, so a ray finishing on an outline reads a wall's grid as it always did.
    if (span.bake_wide > 0u) {
        return baked_profile_distance(span, q);
    }

    // A loop is read through the distance it already answers, moved and folded -- walked,
    // since the span has no bake by here.
    if (closed > 0.5) {
        return abs(profile_distance(chain, q) - middle) - half;
    }

    if (span.count < 2u) {
        return 1e30;
    }
    let runs = span.count - 1u;
    // A round-ended wall about its curve is a chain of capsules: the curve's own distance,
    // ends and all, less half the thickness.
    let capsules = facing == 0.0 && round > 0.5;

    // The nearest foot with `q` on the wall's side of it, and the nearest with `q` on the other.
    var with_wall = 1e30;
    var against = 1e30;
    var past_previous = false;
    var before = vec2<f32>(1.0, 0.0);
    for (var i = 0u; i < runs; i = i + 1u) {
        let a = curve_points[span.start + i];
        let b = curve_points[span.start + i + 1u].xy;
        let step = b - a.xy;
        let len = length(step);
        let dir = step / len;
        let rel = q - a.xy;
        let along = dot(rel, dir);
        let code = u32(a.z + 0.5);
        if (along < 0.0) {
            if (i > 0u && past_previous) {
                // In the joint's fan, on the side the fan is on whichever side of either run.
                let fan = select(-1.0, 1.0, (code & 4u) != 0u);
                if (fan * facing >= 0.0) {
                    with_wall = min(with_wall, length(rel));
                } else {
                    against = min(against, length(rel));
                }
            } else if (i == 0u && capsules) {
                with_wall = min(with_wall, length(rel));
            }
        } else if (along <= len) {
            let lateral = wall_cross(dir, rel);
            var cut_away = false;
            if (lateral * facing > 0.0) {
                if ((code & 1u) != 0u) {
                    cut_away = wall_cut_short(
                        lateral,
                        facing * wall_cross(before, dir) * along,
                        facing * wall_cross(before, rel),
                        thickness
                    );
                }
                if (!cut_away && (code & 2u) != 0u) {
                    let after = normalize(curve_points[span.start + i + 2u].xy - b);
                    cut_away = wall_cut_short(
                        lateral,
                        facing * wall_cross(dir, after) * (len - along),
                        facing * wall_cross(after, q - b),
                        thickness
                    );
                }
            }
            if (!cut_away) {
                if (lateral * facing >= 0.0) {
                    with_wall = min(with_wall, abs(lateral));
                } else {
                    against = min(against, abs(lateral));
                }
            }
        }
        past_previous = along > len;
        before = dir;
    }
    let last = curve_points[span.start + runs].xy;
    if (past_previous && capsules) {
        with_wall = min(with_wall, length(q - last));
    }

    var fibres = with_wall - half;
    if (facing != 0.0) {
        fibres = min(abs(with_wall - half) - half, against);
    }
    if (capsules) {
        return fibres;
    }

    // A round end on a one-sided wall: a disc on the middle of each end fibre. `before` is the
    // last run's direction here, which is the one the far end is read along.
    var disc = 1e30;
    var rounds = 1e30;
    if (facing != 0.0 && round > 0.5) {
        let first = curve_points[span.start];
        let second = curve_points[span.start + 1u].xy;
        let first_code = u32(first.z + 0.5);
        let start = wall_round_end(
            q,
            first.xy,
            normalize(second - first.xy),
            -1.0,
            middle,
            half,
            length(second - first.xy) >= half && (first_code & 3u) == 0u
        );
        let final_start = curve_points[span.start + runs - 1u];
        let final_code = u32(final_start.z + 0.5);
        let end = wall_round_end(
            q,
            last,
            before,
            1.0,
            middle,
            half,
            length(last - final_start.xy) >= half && (final_code & 3u) == 0u
        );
        disc = min(start.x, end.x);
        rounds = min(start.y, end.y);
    }

    // Outside, the feet are the whole answer but for the bare end fibres.
    if (fibres > 0.0 && disc > 0.0) {
        return min(min(fibres, wall_nearest_stretch(faces, q)), disc);
    }

    // Inside, the depth is to the boundary itself: every bare flat edge, and every fan's round,
    // whether or not `q` is in that fan.
    before = normalize(curve_points[span.start + 1u].xy - curve_points[span.start].xy);
    for (var j = 1u; j < runs; j = j + 1u) {
        let at = curve_points[span.start + j];
        let after = normalize(curve_points[span.start + j + 1u].xy - at.xy);
        let fan = select(-1.0, 1.0, (u32(at.z + 0.5) & 4u) != 0u);
        let reach = select(-lo, hi, fan > 0.0);
        if (reach > 0.0) {
            let rel = q - at.xy;
            if (dot(rel, before) >= 0.0 && dot(rel, after) <= 0.0) {
                rounds = min(rounds, abs(length(rel) - reach));
            } else {
                rounds = min(
                    rounds,
                    min(
                        length(rel - wall_left(before) * (fan * reach)),
                        length(rel - wall_left(after) * (fan * reach))
                    )
                );
            }
        }
        before = after;
    }
    return -min(min(wall_nearest_stretch(faces, q), wall_nearest_stretch(sides, q)), rounds);
}

// A wall given thickness along its plane's normal: a thin extrude. `extrude_distance`, asked of
// a wall instead of a profile, and through the same `extrude_over`.
fn wall_extrude_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    thickness: f32,
    facing: f32,
    round: f32,
    closed: f32,
) -> f32 {
    return wall_extrude_rim_distance(
        chain, faces, sides, p, dir, depth, thickness, facing, round, closed, NO_RIM
    );
}

fn wall_extrude_rim_distance(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    thickness: f32,
    facing: f32,
    round: f32,
    closed: f32,
    rim: vec4<f32>,
) -> f32 {
    let along = abs(dir.z);
    if (along < 1e-6) {
        return 1e30;
    }
    let t = p.z / dir.z;
    let landed = p - dir * t;
    return extrude_over_rim(
        wall_distance(chain, faces, sides, landed.xy, thickness, facing, round, closed),
        t,
        dir,
        depth,
        rim
    );
}

// Its outward normal, in world coordinates. Differenced, for the reason `extrude_normal` is.
fn wall_extrude_normal(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    thickness: f32,
    facing: f32,
    round: f32,
    closed: f32,
    euler: vec3<f32>,
) -> vec3<f32> {
    return wall_extrude_rim_normal(
        chain, faces, sides, p, dir, depth, thickness, facing, round, closed, euler, NO_RIM
    );
}

fn wall_extrude_rim_normal(
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    dir: vec3<f32>,
    depth: ExtrudeDepth,
    thickness: f32,
    facing: f32,
    round: f32,
    closed: f32,
    euler: vec3<f32>,
    rim: vec4<f32>,
) -> vec3<f32> {
    // In one loop and not six calls, as the wedge's is and for its reason.
    let h = 1e-3;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let axis = k / 2u;
        let weight = 1.0 - 2.0 * f32(k % 2u);
        var step = vec3<f32>(0.0, 0.0, 0.0);
        step[axis] = h * weight;
        n[axis] += weight * wall_extrude_rim_distance(chain, faces, sides, p + step, dir, depth, thickness, facing, round, closed, rim);
    }
    return normalize(euler_rotation(euler) * n);
}

// Distance to a swept round tube around curve `index`.
//
// Evaluated from the spine buffer rather than unrolled into the shader, so a curve with a
// thousand segments does not become a thousand lines of generated code, and editing a curve
// updates a buffer instead of forcing a recompile.
// A profile carried along a curve: the pipe's walk keeping which segment won, a frame at that
// place, and the extrude's expression in that frame. The same arithmetic `Swept::distance`
// does on the other side, and it has to stay the same: the arc length accumulates through the
// walk; the tangent at a sample is the mean of its two segments' directions
// (`sample_tangents`); the normal is the uploaded frame interpolated within the segment and
// re-squared to the tangent (`SpineFrame::at`); the slab, the corner and the division by the
// stretch are as the CPU takes them. `spine` and `frame` are read in step -- sample `i` of one
// is sample `i` of the other -- which is what `CurveUse::Frame` promises.
// The nearest point of a spine to `p`, by segment: which segment, how far along it, the point
// itself, and the squared distance to it.
struct SegmentHit {
    i: u32,
    t: f32,
    q: vec3<f32>,
    d2: f32,
}

// One segment, tested exactly as the linear walk always tested it. In one place so that every
// caller and the tree agree to the bit -- and so that `gpu_matches_cpu`, which held the walk
// to the CPU's answer, holds the tree to the same one.
fn segment_hit(span: CurveSpan, i: u32, p: vec3<f32>) -> SegmentHit {
    let a = curve_points[span.start + i].xyz;
    let b = curve_points[span.start + i + 1u].xyz;
    let ab = b - a;
    let denom = dot(ab, ab);
    var t = 0.0;
    if denom > 1e-18 {
        t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    }
    let q = a + ab * t;
    return SegmentHit(i, t, q, dot(p - q, p - q));
}

// Strictly nearer, or as near and earlier along the curve. The walk found the earliest of two
// equidistant segments because it compared strictly and visited them in order; the tree visits
// them in whatever order its boxes dictate, so the rule has to be said.
fn nearer(hit: SegmentHit, best: SegmentHit) -> bool {
    return hit.d2 < best.d2 || (hit.d2 == best.d2 && hit.i < best.i);
}

// The nearest segment of a spine to `p`.
//
// **One index over the spine, serving the pipe, the sweep and the attractor** (loop spec P1,
// plan step 6). Every one of them used to walk every segment on every reading: 37 for a pipe
// at the viewport's degree of turn, 154 for a swept glyph, and a pattern multiplied that by the
// instances whose windows overlap a point. The tree is built on the CPU at upload over runs of
// consecutive segments, so a node's box is a stretch of the curve; a node whose box lies no
// nearer than the best segment found so far cannot hold a nearer one and is skipped. Exact,
// because the distance to a box is a lower bound on the distance to anything in it.
//
// **Stackless, and that was measured rather than chosen.** The first form kept an explicit
// stack in a local array, which WGSL spills to slow memory, and its walk diverged across a
// wavefront: the swept glyph went from 154 to 228 ms a frame and a pipe from 34 to 45. The
// nodes are laid out depth first and each carries the index of the node after its subtree, so
// pruning is one jump and the walk is a plain loop with no stack at all.
//
// The pruning test carries a millionth of slack: the box distance and a segment's distance are
// different arithmetic, and a node that is exactly as near as the best segment must still be
// opened for the tie rule above to see what is in it.
fn nearest_segment(span: CurveSpan, p: vec3<f32>) -> SegmentHit {
    var best = SegmentHit(0u, 0.0, curve_points[span.start].xyz, 1e30);
    if span.count < 2u {
        return best;
    }
    if span.bvh_count == 0u {
        for (var i = 0u; i + 1u < span.count; i = i + 1u) {
            let hit = segment_hit(span, i, p);
            if nearer(hit, best) {
                best = hit;
            }
        }
        return best;
    }
    // **First a bound, then the walk.** The walk below visits the curve from its start, and on
    // the way to the nearest stretch every box it meets is nearer than the best found so far,
    // so nothing is ruled out until it gets there: measured, a 1381-segment pipe cost 217 ms
    // against the plain walk's 259, where an index should have made it cost what 173 segments
    // do. So descend once, greedily, toward whichever child's box is nearer, to a single leaf.
    // That leaf is usually the nearest stretch or beside it, its segments give a tight bound
    // for a handful of box tests, and the walk then rules out nearly everything else. No stack:
    // a node's first child is the node after it, and its second is the node after the first
    // child's subtree, which the first child's `w` already names. The greedy leaf's segments
    // are met again by the walk, where the tie rule leaves them as they are.
    var g = 0u;
    loop {
        let ghi = curve_points[span.bvh_start + 2u * g + 1u];
        if ghi.w >= 0.0 {
            let packed = u32(ghi.w);
            let first = packed >> 4u;
            let count = packed & 15u;
            for (var k = 0u; k < count; k = k + 1u) {
                let hit = segment_hit(span, first + k, p);
                if nearer(hit, best) {
                    best = hit;
                }
            }
            break;
        }
        let left = g + 1u;
        let llo = curve_points[span.bvh_start + 2u * left];
        let lhi = curve_points[span.bvh_start + 2u * left + 1u];
        let right = u32(llo.w);
        if right >= span.bvh_count {
            break;
        }
        let rlo = curve_points[span.bvh_start + 2u * right];
        let rhi = curve_points[span.bvh_start + 2u * right + 1u];
        if box_distance(p, rlo.xyz, rhi.xyz) < box_distance(p, llo.xyz, lhi.xyz) {
            g = right;
        } else {
            g = left;
        }
    }

    var n = 0u;
    loop {
        if n >= span.bvh_count {
            break;
        }
        let lo = curve_points[span.bvh_start + 2u * n];
        let hi = curve_points[span.bvh_start + 2u * n + 1u];
        let outside = max(box_distance(p, lo.xyz, hi.xyz), 0.0);
        if outside * outside > best.d2 * (1.0 + 1e-6) {
            // Past this whole stretch of curve: `lo.w` is the node after its subtree.
            n = u32(lo.w);
            continue;
        }
        if hi.w >= 0.0 {
            // A leaf: its first segment and count packed in `hi.w`.
            let packed = u32(hi.w);
            let first = packed >> 4u;
            let count = packed & 15u;
            for (var k = 0u; k < count; k = k + 1u) {
                let hit = segment_hit(span, first + k, p);
                if nearer(hit, best) {
                    best = hit;
                }
            }
        }
        n = n + 1u;
    }
    return best;
}

fn sweep_tangent(span: CurveSpan, i: u32) -> vec3<f32> {
    var before = vec3<f32>(0.0);
    var has_before = false;
    if i > 0u {
        let d = curve_points[span.start + i].xyz - curve_points[span.start + i - 1u].xyz;
        let l = length(d);
        if l > 1e-12 {
            before = d / l;
            has_before = true;
        }
    }
    var after = vec3<f32>(0.0);
    var has_after = false;
    if i + 1u < span.count {
        let d = curve_points[span.start + i + 1u].xyz - curve_points[span.start + i].xyz;
        let l = length(d);
        if l > 1e-12 {
            after = d / l;
            has_after = true;
        }
    }
    if has_before && has_after {
        let m = before + after;
        let l = length(m);
        if l > 1e-12 {
            return m / l;
        }
        // A hairpin: the mean cancels, and the CPU takes the segment after.
        return after;
    }
    if has_after {
        return after;
    }
    return before;
}

// One reading of a sweep, taken against a segment of the spine that has already been chosen.
//
// `ua` and `vb` are the profile's own two axes in the world, and `held` says what to do with
// them: 0 holds the section square to the path and ignores them, 1 starts flush with the
// profile's plane and carries that tilt by decomposing them in the frame where the path starts,
// 2 holds them as they are so the section never turns. Mode 0 reduces every line below to what
// this function was before held sections existed.
//
// `bend` is what the spine's bend costs, as `sweep_over` reads it: the stretch, the section's
// reach, and the tube about the spine the section's own reading is trusted in -- nought where it
// is trusted nowhere.
//
// **Which segment is nearest is a separate question from what the reading is, because they cost
// differently.** Finding the segment walks the spine's index, and the driver unrolls that whole
// walk into every call site; everything after it is a dozen lines of arithmetic. A value and a
// central-differenced normal need seven readings and only one walk, which is what
// `sweep_value_normal` below does with this.
fn sweep_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
) -> f32 {
    return sweep_rim_on(span, fspan, best_i, profile, p, closed, bend, ua, vb, held, NO_RIM);
}

fn sweep_rim_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    rim: vec4<f32>,
) -> f32 {
    let at = sweep_landing(span, fspan, best_i, p, ua, vb, held);
    if at.square == 0.0 {
        return 1e30;
    }
    return sweep_rim_over(profile_distance(profile, at.q), at, closed, bend, rim);
}

// Where a point lands in the section it belongs to: its two coordinates there, how squarely the
// path crosses that section, and how far past either end of the path it is. `square` is nought
// for a path running along its own section, which has no sweep to measure.
struct SweepLanding {
    q: vec2<f32>,
    square: f32,
    slab: f32,
    // How far past the middle of the path, positive toward its end -- what a rim on one end
    // reads to tell which end is nearer. A law reads its own window's middle instead.
    toward: f32,
    // How far along the path the point landed, in millimetres of arc, unclamped past either
    // end. The slab above is this against the whole path; a law reads it against its own
    // window, and the section's scale and turn are functions of it.
    w: f32,
    // How far the point is from the spine, which is what the tube is measured against.
    near: f32,
    // How far the frame of the walked spine leans off its segments, as `1 - cos`; the range of
    // its twist about them, or -1 where it does not meet itself round a closed spine; and how
    // far out the first segment's own region closes over it: the entry
    // `fieldform_field::sweep_frame` puts after the frame's samples, because it belongs to this
    // walk and not to the curve.
    tilt: vec3<f32>,
}

// **Everything about a sweep but what is read in the section**, which is what lets a profile
// and a wall be carried by one walk: `sweep_on` reads a profile at `q` and `wall_sweep_on` a
// wall, and both hand the reading to `sweep_over`. The arithmetic and its order are what
// `sweep_on` did in one piece before a sweep could carry a wall.
fn sweep_landing(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    p: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
) -> SweepLanding {
    // The nearest point on that segment, and its arc from the uploaded cumulative lengths: the
    // walk used to sum segment lengths as it went, and a tree visits them in no such order.
    // Identical arithmetic to what `nearest_segment` returns for the segment it chose, which is
    // what keeps `sweep_distance` below the same value it has always been.
    let hit = segment_hit(span, best_i, p);
    let best_t = hit.t;
    let best_q = hit.q;
    let seg = curve_points[span.start + best_i + 1u].xyz - curve_points[span.start + best_i].xyz;
    let best_arc = curve_points[span.start + best_i].w + length(seg) * best_t;
    let arc = curve_points[span.start + span.count - 1u].w;
    // The frame within the winning segment, as `SpineFrame::at` reads it.
    let j = min(best_i + 1u, span.count - 1u);
    let t0 = sweep_tangent(span, best_i);
    let t1 = sweep_tangent(span, j);
    var tan = t0 * (1.0 - best_t) + t1 * best_t;
    let tl = length(tan);
    if tl > 1e-12 {
        tan = tan / tl;
    } else {
        tan = t0;
    }
    let n0 = curve_points[fspan.start + best_i].xyz;
    let n1 = curve_points[fspan.start + j].xyz;
    var nor = n0 * (1.0 - best_t) + n1 * best_t;
    nor = nor - tan * dot(nor, tan);
    let nl = length(nor);
    if nl > 1e-12 {
        nor = nor / nl;
    } else {
        nor = n0;
    }
    let bin = cross(tan, nor);
    // The section's own axes.
    var u = nor;
    var v = bin;
    if held > 1.5 {
        // Parallel: the profile's own, never turning.
        u = ua;
        v = vb;
    } else if held > 0.5 {
        // Flush: the profile's own pair written in the frame where the path *starts*, then
        // rebuilt in the frame here — which is what carries the tilt along. Decomposed against
        // this shader's own frame rather than handed across, because the two sides flatten the
        // curve differently and measuring one's frame against the other's moved the solid.
        let t0 = sweep_tangent(span, 0u);
        let n0 = curve_points[fspan.start].xyz;
        let b0 = cross(t0, n0);
        let a = vec3<f32>(dot(ua, t0), dot(ua, n0), dot(ua, b0));
        let b = vec3<f32>(dot(vb, t0), dot(vb, n0), dot(vb, b0));
        u = tan * a.x + nor * a.y + bin * a.z;
        v = tan * b.x + nor * b.y + bin * b.z;
    }
    let m = cross(u, v);
    let d = p - best_q;
    // **Slid onto the section, not projected across it** — the leaning extrude's arithmetic
    // (`extrude_distance` above), with the spine's frame in place of the plane's. The section a
    // point belongs to is the one that contains it, which for a held section is not the one
    // through its nearest spine point.
    let square = dot(tan, m);
    let tilt = curve_points[fspan.start + span.count].xyz;
    if abs(square) <= 1e-9 {
        return SweepLanding(vec2<f32>(0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, tilt);
    }
    let slide = dot(d, m) / square;
    let landed = d - tan * slide;
    // Beyond an end the nearest point is clamped to it and the slide carries the overshoot;
    // adding it back gives the unclamped arc the slab needs — and it tilts the end caps with
    // the section, which is what puts the start face in the profile's own plane.
    let w = best_arc + slide;
    let toward = w - arc * 0.5;
    let slab = abs(toward) - arc * 0.5;
    return SweepLanding(
        vec2<f32>(dot(landed, u), dot(landed, v)), square, slab, toward, w, length(d), tilt
    );
}

// How a swept section changes along its path -- `fieldform_field::SectionLaw`, as six numbers.
//
// `kind` is 0 for no taper, 1 for a scale and 2 for a draft, and `amount` is the end scale or
// the draft's tangent; `shape` is 0 linear, 1 smoothstep, 2 smootherstep, exactly as `ramp`
// reads it. `a0` and `a1` are the window in millimetres of arc. **Six slots rather than six
// literals**, because every one of them is a number somebody drags, and a literal would
// rebuild the shader at the frame rate.
struct SweepLaw {
    kind: f32,
    amount: f32,
    twist: f32,
    shape: f32,
    a0: f32,
    a1: f32,
    // Nought where a draft rounds the section's corners and one where it mitres them. Read
    // only by a draft; a scale moves no wall along its own normal and leaves every corner as
    // it was drawn.
    sharp: f32,
}

// How far along the window a point at arc `w` is, clamped: past either end the law holds the
// value it reached there, so each cap is a flat section of the state the law was in.
fn law_fraction(law: SweepLaw, w: f32) -> f32 {
    let run = law.a1 - law.a0;
    if run <= 0.0 {
        return 0.0;
    }
    return clamp((w - law.a0) / run, 0.0, 1.0);
}

// The shape's own easing, the same three `ramp` has and for the same reason.
fn law_eased(shape: f32, t: f32) -> f32 {
    if (shape > 1.5) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }
    if (shape > 0.5) {
        return t * t * (3.0 - 2.0 * t);
    }
    return t;
}

// Where in the drawn section a point at `q` and arc `w` is read, once the law has turned the
// section: `R(-theta) q`. `sweep-v2-spec.md` section 2's one mechanism, with the scale applied
// by the caller because the reading has to be multiplied back by it.
fn law_point(law: SweepLaw, q: vec2<f32>, w: f32) -> vec2<f32> {
    let theta = law.twist * law_eased(law.shape, law_fraction(law, w));
    if theta == 0.0 {
        return q;
    }
    let c = cos(-theta);
    let s = sin(-theta);
    return vec2<f32>(c * q.x - s * q.y, s * q.x + c * q.y);
}

// How big the section is at arc `w`, as a multiple of the drawn one. `s * outline(q / s)` is
// the exact distance to the scaled outline, so a scaled section is as exact as the section was.
fn law_scale(law: SweepLaw, w: f32) -> f32 {
    if law.kind < 0.5 || law.kind > 1.5 {
        return 1.0;
    }
    return 1.0 + (law.amount - 1.0) * law_eased(law.shape, law_fraction(law, w));
}

// How far out of the section a draft has pushed its walls by arc `w`: the outline's
// `(w - a0) tan(alpha)` contour, which is what the extrude's own draft is. Unclamped past the
// ends, as the extrude's is -- the contour keeps growing and the slab is what cuts it.
fn law_offset(law: SweepLaw, w: f32) -> f32 {
    if law.kind < 1.5 {
        return 0.0;
    }
    return (w - law.a0) * law.amount;
}

// Whether the law leaves the section's walls square to its caps. A drafted wall leans, so the
// corner between the two is not Pythagoras -- the same guard `extrude_distance` has.
fn law_upright(law: SweepLaw) -> bool {
    return law.kind < 1.5 || law.amount == 0.0;
}

// The slab over the law's own window rather than the whole path.
fn law_slab(law: SweepLaw, w: f32) -> f32 {
    return abs(w - (law.a0 + law.a1) * 0.5) - (law.a1 - law.a0) * 0.5;
}

// A section's reading where the point landed, made the sweep's: shortened by the lean, capped
// at the path's ends, and divided by the bend's stretch -- all of it, or inside the tube only
// the caps read from inside (`sweep_in_tube`).
fn sweep_over(
    section: f32,
    at: SweepLanding,
    closed: f32,
    bend: vec3<f32>,
) -> f32 {
    return sweep_rim_over(section, at, closed, bend, NO_RIM);
}

fn sweep_rim_over(
    section: f32,
    at: SweepLanding,
    closed: f32,
    bend: vec3<f32>,
    rim: vec4<f32>,
) -> f32 {
    // Shortened by how far the section leans, because a reading taken in a leaning section
    // overstates by `1 / cos`, and overstating is what the contract forbids.
    let lean = abs(at.square);
    if bend.z > 0.0 {
        return sweep_in_tube_rim(
            section, lean, at.slab, at.toward, lean >= 1.0 - 1e-6, at, closed, bend, rim
        );
    }
    if closed > 0.5 {
        return section * lean / bend.x;
    }
    // A rim through the blend's bound, which holds whatever the lean does to the corner.
    if rim.x > 0.0 {
        return rim_join_ends(section * lean, at.slab, at.toward, rim) / bend.x;
    }
    // Pythagoras only between perpendicular measurements: a leaning section and its own end cap
    // are not, and taking the corner anyway overstated a 45-degree lean by 8 mm where
    // `SkewExtrusion` first met it.
    if section > 0.0 && at.slab > 0.0 && lean >= 1.0 - 1e-6 {
        return length(vec2<f32>(section, at.slab)) / bend.x;
    }
    return max(section * lean, at.slab) / bend.x;
}

// The same as `sweep_over` under a law: the slab is the window's, and a drafted section does
// not take the corner, for the reason given at `law_upright`. A law that moves the section has
// no tube; one whose numbers have all come to rest has, and reads as the plain sweep does.
fn sweep_law_over(
    section: f32,
    at: SweepLanding,
    law: SweepLaw,
    closed: f32,
    bend: vec3<f32>,
) -> f32 {
    return sweep_law_rim_over(section, at, law, closed, bend, NO_RIM);
}

fn sweep_law_rim_over(
    section: f32,
    at: SweepLanding,
    law: SweepLaw,
    closed: f32,
    bend: vec3<f32>,
    rim: vec4<f32>,
) -> f32 {
    let lean = abs(at.square);
    let slab = law_slab(law, at.w);
    let toward = at.w - (law.a0 + law.a1) * 0.5;
    if bend.z > 0.0 {
        return sweep_in_tube_rim(
            section, lean, slab, toward, lean >= 1.0 - 1e-6 && law_upright(law), at, closed, bend
        , rim);
    }
    if closed > 0.5 {
        return section * lean / bend.x;
    }
    if rim.x > 0.0 {
        return rim_join_ends(section * lean, slab, toward, rim) / bend.x;
    }
    if section > 0.0 && slab > 0.0 && lean >= 1.0 - 1e-6 && law_upright(law) {
        return length(vec2<f32>(section, slab)) / bend.x;
    }
    return max(section * lean, slab) / bend.x;
}

// **Inside the tube the section reads as it is** -- `fieldform_field::Swept::distance`, and
// `sweep-spec.md` section 4. The frame turns about no axis of its own, so the section's two
// coordinates change at the rate the point moves; the bend stretches only the arc, and only the
// slab reads it. Past an end the map is rigid, so a positive slab is exact, and a negative one
// is read inside the section, within the reach, where the stretch bounds it.
//
// **Less what the frame's lean and twist can account for** (section 4.1). The walked
// polyline's frame leans off each segment by up to half a joint's turn, which shortens a reading
// `near` from the spine by up to `(1 - cos) near` against a rigid segment's frame, and the solid
// reaches `reach / cos`; off a plane it also twists about the segments. So the reading gives up
// the lean's share here and at the surface and the twist's at the surface, and where that leaves
// less than dividing does, or the frame does not meet itself round a closed spine, the divided
// reading stands. The tube is decided again against the lengthened reach and narrowed to where
// the walk's first segment closes over, since a walk leans further than the curve's own
// flattening did and is cut differently.
//
// And no reading claims more than the tube allows: every point within it has one nearest place
// along the spine, so a reading whose way to the surface stays inside is held as above, and one
// whose way would leave it is held to what is known without it -- at least `tube - near` to the
// surface, or the reading would have been held, and at least `near - reach`.
fn sweep_in_tube(
    section: f32,
    lean: f32,
    slab: f32,
    corner: bool,
    at: SweepLanding,
    closed: f32,
    bend: vec3<f32>,
) -> f32 {
    return sweep_in_tube_rim(section, lean, slab, 0.0, corner, at, closed, bend, NO_RIM);
}

fn sweep_in_tube_rim(
    section: f32,
    lean: f32,
    slab: f32,
    toward: f32,
    corner: bool,
    at: SweepLanding,
    closed: f32,
    bend: vec3<f32>,
    rim: vec4<f32>,
) -> f32 {
    var own = section * lean;
    var divided = section * lean / bend.x;
    if closed <= 0.5 {
        if rim.x > 0.0 {
            divided = rim_join_ends(section * lean, slab, toward, rim) / bend.x;
        } else if section > 0.0 && slab > 0.0 && corner {
            divided = length(vec2<f32>(section, slab)) / bend.x;
        } else {
            divided = max(section * lean, slab) / bend.x;
        }
        var cap = slab;
        if cap <= 0.0 {
            cap = cap / bend.x;
        }
        // The hard corner's even when the rims are rounded: dividing only the negative slab
        // would move a rim's arc a stretch along the path, and the rounded solid lies inside
        // the hard one, so outside it this less its allowance still claims no more than there
        // is (`Swept::distance`).
        if section > 0.0 && cap > 0.0 && corner {
            own = length(vec2<f32>(section, cap));
        } else {
            own = max(section * lean, cap);
        }
    }
    // `tube * cos > reach` rather than a division, which keeps a lean of a right angle or more
    // -- a cosine of nought or less -- out of the tube without dividing by it.
    let tube = min(bend.z, at.tilt.z);
    let cos_lean = 1.0 - at.tilt.x;
    if tube * cos_lean <= bend.y {
        return divided;
    }
    let reach = bend.y / cos_lean;
    var reading = divided;
    if at.tilt.y >= 0.0 {
        let allowance = at.tilt.x * (at.near + reach) + reach * at.tilt.y;
        if own > 0.0 {
            reading = max(divided, own - allowance);
        } else if rim.x <= 0.0 {
            // Inside, the hard solid's depth says nothing of a rounded one's.
            reading = min(divided, own + allowance);
        }
    }
    return min(reading, max(at.near - reach, tube - at.near));
}

// One reading of a sweep under a law: the landing the plain one makes, with the section scaled,
// turned and offset where the point landed. **The landing is shared rather than copied**, which
// is the whole reason the law is a separate pair of functions and not a separate sweep: the
// walk of the spine's index is what the driver unrolls into every call site, and there is one
// of it.
fn sweep_law_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
) -> f32 {
    return sweep_law_rim_on(span, fspan, best_i, profile, miters, p, closed, bend, ua, vb, held, law, NO_RIM);
}

fn sweep_law_rim_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let at = sweep_landing(span, fspan, best_i, p, ua, vb, held);
    if at.square == 0.0 {
        return 1e30;
    }
    let s = law_scale(law, at.w);
    let q = law_point(law, at.q, at.w);
    var section: f32;
    if (law.kind > 1.5 && law.sharp > 0.5) {
        // A mitred draft: the section's own offset, which no arithmetic on a reading can give.
        // A scale and a draft are never both set, so `s` is one here.
        section = profile_mitred(profile, miters, q, law_offset(law, at.w));
    } else {
        section = s * profile_distance(profile, q / s) - law_offset(law, at.w);
    }
    return sweep_law_rim_over(section, at, law, closed, bend, rim);
}

fn sweep_law_distance(
    spine: u32,
    frame: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
) -> f32 {
    return sweep_law_rim_distance(spine, frame, profile, miters, p, closed, bend, ua, vb, held, law, NO_RIM);
}

fn sweep_law_rim_distance(
    spine: u32,
    frame: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return sweep_law_rim_on(
        span, curve_spans[frame], best_i, profile, miters, p, closed, bend, ua, vb, held, law
    , rim);
}

// Value and normal from one walk, as `sweep_value_normal` is and for the measured reason there.
fn sweep_law_value_normal(
    spine: u32,
    frame: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
) -> vec4<f32> {
    return sweep_law_rim_value_normal(spine, frame, profile, miters, p, closed, bend, ua, vb, held, law, NO_RIM);
}

fn sweep_law_rim_value_normal(
    spine: u32,
    frame: u32,
    profile: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    law: SweepLaw,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let h = 1e-3;
    var value = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 7u; k = k + 1u) {
        var step = vec3<f32>(0.0, 0.0, 0.0);
        var weight = 0.0;
        if k > 0u {
            let axis = (k - 1u) / 2u;
            weight = 1.0 - 2.0 * f32((k - 1u) % 2u);
            step[axis] = h * weight;
        }
        // The value with the tube, the normal off the divided reading: see `sweep_value_normal`.
        let reading = sweep_law_rim_on(
            span, fspan, best_i, profile, miters, p + step, closed,
            select(vec3<f32>(bend.x, bend.y, 0.0), bend, k == 0u), ua, vb, held, law
        , rim);
        if k == 0u {
            value = reading;
        } else {
            n[(k - 1u) / 2u] += weight * reading;
        }
    }
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

// The same reading with a wall in the section where `sweep_on` has a profile: an open curve
// given a thickness, carried along the path. One landing, and a different thing read there.
fn wall_sweep_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
) -> f32 {
    return wall_sweep_rim_on(span, fspan, best_i, chain, faces, sides, p, closed, bend, ua, vb, held, wall, NO_RIM);
}

fn wall_sweep_rim_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    rim: vec4<f32>,
) -> f32 {
    let at = sweep_landing(span, fspan, best_i, p, ua, vb, held);
    if at.square == 0.0 {
        return 1e30;
    }
    return sweep_rim_over(
        wall_distance(chain, faces, sides, at.q, wall.x, wall.y, wall.z, wall.w),
        at,
        closed,
        bend
    , rim);
}

// A wall's normal in `xyz` and its value in `w`, carried along a path: `sweep_value_normal`
// with a wall in the section. `wall` is its four numbers together -- thickness, which way the
// band lies, whether the ends are round, whether the chain closes -- in the order
// `wall_distance` takes them.
//
// **The seven readings are one loop and not seven calls**, which `sweep_value_normal` is not:
// a wall is the heaviest thing the prelude reads, and the driver inlines each call where it is
// written, so seven of them are seven copies of the walk in one function. The offsets come out
// of the index, so there is one copy and it runs seven times.
fn wall_sweep_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
) -> vec4<f32> {
    return wall_sweep_rim_value_normal(spine, frame, chain, faces, sides, p, closed, bend, ua, vb, held, wall, NO_RIM);
}

fn wall_sweep_rim_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let h = 1e-3;
    var value = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    // Reading nought is at the point itself; one to six step out along +x, -x, +y, -y, +z, -z.
    for (var k = 0u; k < 7u; k = k + 1u) {
        var step = vec3<f32>(0.0, 0.0, 0.0);
        var weight = 0.0;
        if k > 0u {
            let axis = (k - 1u) / 2u;
            weight = 1.0 - 2.0 * f32((k - 1u) % 2u);
            step[axis] = h * weight;
        }
        // The value with the tube, the normal off the divided reading: see `sweep_value_normal`.
        let reading = wall_sweep_rim_on(
            span, fspan, best_i, chain, faces, sides, p + step, closed,
            select(vec3<f32>(bend.x, bend.y, 0.0), bend, k == 0u), ua, vb, held, wall
        , rim);
        if k == 0u {
            value = reading;
        } else {
            n[(k - 1u) / 2u] += weight * reading;
        }
    }
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

fn wall_sweep_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
) -> f32 {
    return wall_sweep_rim_distance(spine, frame, chain, faces, sides, p, closed, bend, ua, vb, held, wall, NO_RIM);
}

fn wall_sweep_rim_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return wall_sweep_rim_on(
        span, curve_spans[frame], best_i, chain, faces, sides, p, closed, bend, ua, vb, held, wall
    , rim);
}

// A wall carried under a law, arm for arm against `Wall::distance_tapered`.
//
// **A loop's wall keeps its thickness and an open chain's does not**, which is the one place
// the two kinds of wall part company here. A wall round a loop is the loop's own distance
// folded about the middle of a band, so scaling the loop and moving the band are each one line
// and the band stays as wide as it was: a tapered tube keeps its wall. An open wall is not a
// function of its curve's distance alone -- its mitres and end faces are built for one
// thickness on one curve -- so it is scaled whole, thickness and all, which is exact and is
// said beside the field. A draft on an open wall is refused by the kernel before anything
// reaches here, so `law_offset` is only ever added to a loop's.
fn wall_sweep_law_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> f32 {
    return wall_sweep_law_rim_on(span, fspan, best_i, chain, faces, sides, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_law_rim_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let at = sweep_landing(span, fspan, best_i, p, ua, vb, held);
    if at.square == 0.0 {
        return 1e30;
    }
    let s = law_scale(law, at.w);
    let q = law_point(law, at.q, at.w) / s;
    // **The wall's own reading, scaled, less the draft's offset.** A draft offsets the *solid*,
    // so a band's two faces both move outward along their own normals and the band grows by the
    // offset on each side — which is what `Wall::offset_by` does and what the extrude's own
    // `extrude_over` does (`wall_distance(...) - grown`).
    //
    // **This moved the band's middle instead**, keeping its thickness, which slides a drafted
    // tube's whole wall outward rather than thickening it. The outer face lands in the same
    // place either way, so every outside view agreed and the whole disagreement was down the
    // bore: measured at 6.9 mm on a 50 mm square loop with a 4 mm wall drafted 2° over
    // 100 mm. The harness had no drafted-wall sweep to catch it (`SweptDrafted` carries a
    // closed profile, the thin sweeps carry a scale), which is why 87 scenes agreed with it in.
    let section = s * wall_distance(chain, faces, sides, q, wall.x, wall.y, wall.z, wall.w)
        - law_offset(law, at.w);
    return sweep_law_rim_over(section, at, law, closed, bend, rim);
}

// **A wall carried under a law whose draft mitres its corners, as its own pair of entry
// points.** The reading differs from `wall_sweep_law_on` in one line, and branching on it
// inside that function instead cost a shader *two* polyline walks where it needs one -- which
// the driver, inlining the body seven times for the normal, could not compile at all: the
// harness crashed the process outright on a thin scaled sweep. Codegen picks the pair; only
// the one picked is ever written into a shader.
fn wall_sweep_sharp_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> f32 {
    return wall_sweep_sharp_rim_on(span, fspan, best_i, chain, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_sharp_rim_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let at = sweep_landing(span, fspan, best_i, p, ua, vb, held);
    if at.square == 0.0 {
        return 1e30;
    }
    let q = law_point(law, at.q, at.w);
    // A loop's wall only: an open one has no inside for a draft to push out from, and is
    // refused before anything reaches here.
    let section = wall_mitred(chain, miters, q, law_offset(law, at.w), wall.x, wall.y);
    return sweep_law_rim_over(section, at, law, closed, bend, rim);
}

fn wall_sweep_sharp_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> f32 {
    return wall_sweep_sharp_rim_distance(spine, frame, chain, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_sharp_rim_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return wall_sweep_sharp_rim_on(
        span, curve_spans[frame], best_i, chain, miters, p, closed, bend, ua, vb, held,
        wall, law
    , rim);
}

fn wall_sweep_sharp_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> vec4<f32> {
    return wall_sweep_sharp_rim_value_normal(spine, frame, chain, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_sharp_rim_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let h = 1e-3;
    var value = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 7u; k = k + 1u) {
        var step = vec3<f32>(0.0, 0.0, 0.0);
        var weight = 0.0;
        if k > 0u {
            let axis = (k - 1u) / 2u;
            weight = 1.0 - 2.0 * f32((k - 1u) % 2u);
            step[axis] = h * weight;
        }
        // The value with the tube, the normal off the divided reading: see `sweep_value_normal`.
        let reading = wall_sweep_sharp_rim_on(
            span, fspan, best_i, chain, miters, p + step, closed,
            select(vec3<f32>(bend.x, bend.y, 0.0), bend, k == 0u), ua, vb, held, wall, law
        , rim);
        if k == 0u {
            value = reading;
        } else {
            n[(k - 1u) / 2u] += weight * reading;
        }
    }
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

fn wall_sweep_law_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> f32 {
    return wall_sweep_law_rim_distance(spine, frame, chain, faces, sides, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_law_rim_distance(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return wall_sweep_law_rim_on(
        span, curve_spans[frame], best_i, chain, faces, sides, miters, p, closed, bend, ua,
        vb, held, wall, law
    , rim);
}

// Seven readings, one walk, one copy of the body -- as `wall_sweep_value_normal` is, and for
// the compile-time reason recorded there.
fn wall_sweep_law_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
) -> vec4<f32> {
    return wall_sweep_law_rim_value_normal(spine, frame, chain, faces, sides, miters, p, closed, bend, ua, vb, held, wall, law, NO_RIM);
}

fn wall_sweep_law_rim_value_normal(
    spine: u32,
    frame: u32,
    chain: u32,
    faces: u32,
    sides: u32,
    miters: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    wall: vec4<f32>,
    law: SweepLaw,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let h = 1e-3;
    var value = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 7u; k = k + 1u) {
        var step = vec3<f32>(0.0, 0.0, 0.0);
        var weight = 0.0;
        if k > 0u {
            let axis = (k - 1u) / 2u;
            weight = 1.0 - 2.0 * f32((k - 1u) % 2u);
            step[axis] = h * weight;
        }
        // The value with the tube, the normal off the divided reading: see `sweep_value_normal`.
        let reading = wall_sweep_law_rim_on(
            span, fspan, best_i, chain, faces, sides, miters, p + step, closed,
            select(vec3<f32>(bend.x, bend.y, 0.0), bend, k == 0u), ua, vb, held, wall, law
        , rim);
        if k == 0u {
            value = reading;
        } else {
            n[(k - 1u) / 2u] += weight * reading;
        }
    }
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

fn sweep_distance(
    spine: u32,
    frame: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
) -> f32 {
    return sweep_rim_distance(spine, frame, profile, p, closed, bend, ua, vb, held, NO_RIM);
}

fn sweep_rim_distance(
    spine: u32,
    frame: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return sweep_rim_on(span, curve_spans[frame], best_i, profile, p, closed, bend, ua, vb, held, rim);
}

// A sweep's normal in `xyz` and its value in `w`, from **one** walk of the spine.
//
// The normal is still central-differenced, for the reason `extrude_normal` gives: a profile is a
// polyline and its gradient is piecewise constant with a discontinuity at every vertex, so there
// is no analytic normal here for either side to derive, and differencing is what the CPU does
// too. What changed is how many times the spine gets walked to produce one.
//
// **Finding the nearest segment is the expensive part, and it is now done once for all seven
// readings.** The driver unrolls the spine's whole index into every call site, and its compile
// time is worse than cubic in how many it has to unroll, so a normal pass that called
// `sweep_distance` six times for the gradient and once more for the value paid for eight walks
// per sweep. Measured on the owner's wing — a sweep, mirrored, blended to a fuselage at G1 — by
// `wing_compile_cost`, on an RTX A500:
//
// | shape of the normal | driver |
// |---|---|
// | eight walks per sweep, as first written | 8419 ms |
// | one walk per reading, seven readings | 4461 ms |
// | one walk for all seven | 3756 ms |
//
// Value and normal together rather than in two calls, because the normal pass wants both at the
// same point — choosing between a boolean's operands needs their values — and asking for them
// separately is what made it two walks where one would do. The same `vec4` layout the generated
// body functions already return.
//
// **Held still across the seven readings: which segment. Recomputed for each of them:
// everything else** — the nearest point on that segment, the frame there, the arc, the section.
// Over a thousandth of a millimetre the nearest segment is the same segment, except exactly
// where two of them are equidistant, and there the distance has a crease and no normal to be
// wrong about.
fn sweep_value_normal(
    spine: u32,
    frame: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
) -> vec4<f32> {
    return sweep_rim_value_normal(spine, frame, profile, p, closed, bend, ua, vb, held, NO_RIM);
}

fn sweep_rim_value_normal(
    spine: u32,
    frame: u32,
    profile: u32,
    p: vec3<f32>,
    closed: f32,
    bend: vec3<f32>,
    ua: vec3<f32>,
    vb: vec3<f32>,
    held: f32,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let value = sweep_rim_on(span, fspan, best_i, profile, p, closed, bend, ua, vb, held, rim);
    // **The normal is read off the divided reading** -- `sweep_over` without the tube. It has
    // the surface's normal wherever the reading in the tube does, and none of the crease where
    // that reading gives way to the divided one, the frame's allowance out from the surface
    // (`sweep-spec.md` section 4.1). Six readings a thousandth of a millimetre apart straddle
    // the crease wherever the march stops about that far out, and its two slopes lean the
    // normal; the same goes for the four functions above.
    let level = vec3<f32>(bend.x, bend.y, 0.0);
    let h = 1e-3;
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    let n = vec3<f32>(
        sweep_rim_on(span, fspan, best_i, profile, p + dx, closed, level, ua, vb, held, rim)
            - sweep_rim_on(span, fspan, best_i, profile, p - dx, closed, level, ua, vb, held, rim),
        sweep_rim_on(span, fspan, best_i, profile, p + dy, closed, level, ua, vb, held, rim)
            - sweep_rim_on(span, fspan, best_i, profile, p - dy, closed, level, ua, vb, held, rim),
        sweep_rim_on(span, fspan, best_i, profile, p + dz, closed, level, ua, vb, held, rim)
            - sweep_rim_on(span, fspan, best_i, profile, p - dz, closed, level, ua, vb, held, rim),
    );
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

// A loft: closed sections passed through in turn along a centerline, interpolated **vertex to
// vertex** -- `fieldform_field::Lofted`, arm for arm. `loft-spec.md` §8 and §13.
//
// Every section was resampled to one vertex count with its corners matched to the others'
// (`correspond`, on the CPU, once) and uploaded as `CurveUse::LoftSection`. The section at a
// station between two listed ones is the polygon whose vertex j lies between their vertex js --
// straight between them at G0, on a Hermite rail through every listed position at G1 -- and the
// reading is the signed distance to that polygon: `profile_distance`'s walk, with each vertex
// made on the way from the two (G0) or four (G1) stored sections' vertex j. **One walk of one
// polygon per reading whatever the smoothness**, where the first design walked two or four
// stored sections of however many points they were drawn with.
//
// **The table is in `params`, not in the source**, ruled by the owner: at `params[table + 2k]`
// the k-th section's station, in millimetres of arc along the uploaded spine, and at
// `params[table + 2k + 1]` the binding its resampled section is read from. A station moves
// whenever anybody drags a section or the centerline, and a number in the source would rebuild
// the shader at the frame rate -- the reason every other dragged number here is a slot.
//
// The landing is the sweep's own, held square: the nearest spine point, the rotation-minimising
// frame there, and the arc carried past either end by the overshoot.
// Three slots a section: its station, its binding, and its rails' slopes' binding -- the
// section's own again at G0, where no slope is read.
fn loft_station(table: u32, k: u32) -> f32 {
    return params[table + 3u * k];
}

fn loft_binding(table: u32, k: u32) -> u32 {
    return u32(params[table + 3u * k + 1u]);
}

fn loft_slope_binding(table: u32, k: u32) -> u32 {
    return u32(params[table + 3u * k + 2u]);
}

// Vertex `j` of section `k`, as uploaded.
fn loft_point(table: u32, k: u32, j: u32) -> vec2<f32> {
    let span = curve_spans[loft_binding(table, k)];
    return curve_points[span.start + j].xy;
}

// The slope of vertex j's rail at station k, per millimetre of arc -- `Lofted::new`'s slopes,
// uploaded beside the section (`CurveUse::LoftSlope`) rather than made here from two more
// points and two stations: four reads a vertex where that was six, measured.
fn loft_rail_slope(table: u32, k: u32, j: u32) -> vec2<f32> {
    let span = curve_spans[loft_slope_binding(table, k)];
    return curve_points[span.start + j].xy;
}

// Vertex j of the section at `t` through span `i` -- `Lofted::vertex`.
fn loft_vertex(table: u32, count: u32, i: u32, t: f32, eased: f32, j: u32) -> vec2<f32> {
    let a = loft_point(table, i, j);
    let b = loft_point(table, i + 1u, j);
    if (eased < 0.5) {
        return a + (b - a) * t;
    }
    let h = loft_station(table, i + 1u) - loft_station(table, i);
    let m0 = loft_rail_slope(table, i, j);
    let m1 = loft_rail_slope(table, i + 1u, j);
    let t2 = t * t;
    let t3 = t2 * t;
    return a * (2.0 * t3 - 3.0 * t2 + 1.0)
        + m0 * ((t3 - 2.0 * t2 + t) * h)
        + b * (-2.0 * t3 + 3.0 * t2)
        + m1 * ((t3 - t2) * h);
}

// The signed distance from `q` to the section at `t` through span `i`: the magnitude the
// nearest segment, the sign crossing parity by the same half-open rule `profile_distance`
// uses, over vertices made on the way. `Lofted::section_distance`, line for line.
fn loft_section_distance(table: u32, count: u32, i: u32, t: f32, eased: f32, q: vec2<f32>) -> f32 {
    let m = curve_spans[loft_binding(table, i)].count;
    if (m < 3u) {
        return 1e30;
    }
    var nearest = 1e30;
    var inside = false;
    var a = loft_vertex(table, count, i, t, eased, m - 1u);
    for (var j = 0u; j < m; j = j + 1u) {
        let b = loft_vertex(table, count, i, t, eased, j);
        let ab = b - a;
        let len2 = dot(ab, ab);
        var s = 0.0;
        if (len2 > 0.0) {
            s = clamp(dot(q - a, ab) / len2, 0.0, 1.0);
        }
        nearest = min(nearest, length(q - (a + ab * s)));
        if ((a.y > q.y) != (b.y > q.y)) {
            let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if (q.x < x) {
                inside = !inside;
            }
        }
        a = b;
    }
    return select(nearest, -nearest, inside);
}

fn loft_on(
    span: CurveSpan,
    fspan: CurveSpan,
    best_i: u32,
    p: vec3<f32>,
    table: u32,
    count: u32,
    eased: f32,
    stretch: f32,
    rim: vec4<f32>,
) -> f32 {
    // Held square: the profile axes are ignored and `held` is nought.
    let at = sweep_landing(
        span, fspan, best_i, p, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0
    );
    if at.square == 0.0 {
        return 1e30;
    }
    let w = at.w;
    let q = at.q;

    // Which span the arc falls in, clamped: past either end the interpolation holds that end's
    // section, and the slab is what cuts the solid.
    var i = 0u;
    for (var k = 1u; k + 1u < count; k = k + 1u) {
        if (w > loft_station(table, k)) {
            i = k;
        }
    }
    let a0 = loft_station(table, i);
    let a1 = loft_station(table, i + 1u);
    let h = a1 - a0;
    let t = select(0.0, clamp((w - a0) / h, 0.0, 1.0), h > 0.0);
    let section = loft_section_distance(table, count, i, t, eased, q);

    // The slab over the stretch the sections occupy, and the corner where the two are square,
    // which for a section held square to its path they always are.
    let first = loft_station(table, 0u);
    let last = loft_station(table, count - 1u);
    let slab = abs(w - 0.5 * (first + last)) - 0.5 * (last - first);
    // A rim through the blend's bound, as `Lofted::distance` takes it.
    if (rim.x > 0.0) {
        return rim_join_ends(section, slab, w - 0.5 * (first + last), rim) / stretch;
    }
    if (section > 0.0 && slab > 0.0) {
        return length(vec2<f32>(section, slab)) / stretch;
    }
    return max(section, slab) / stretch;
}

fn loft_distance(
    spine: u32,
    frame: u32,
    p: vec3<f32>,
    table: u32,
    count: u32,
    eased: f32,
    stretch: f32,
) -> f32 {
    return loft_rim_distance(spine, frame, p, table, count, eased, stretch, NO_RIM);
}

fn loft_value_normal(
    spine: u32,
    frame: u32,
    p: vec3<f32>,
    table: u32,
    count: u32,
    eased: f32,
    stretch: f32,
) -> vec4<f32> {
    return loft_rim_value_normal(spine, frame, p, table, count, eased, stretch, NO_RIM);
}

// The loft with its rims rounded or chamfered where the side meets the first and the last
// section -- `rim` the reach and the exponent, `NO_RIM` from the plain names above, which
// are what a loft without rims compiles to.
fn loft_rim_distance(
    spine: u32,
    frame: u32,
    p: vec3<f32>,
    table: u32,
    count: u32,
    eased: f32,
    stretch: f32,
    rim: vec4<f32>,
) -> f32 {
    let span = curve_spans[spine];
    if span.count < 2u {
        return 1e30;
    }
    let best_i = nearest_segment(span, p).i;
    return loft_on(span, curve_spans[frame], best_i, p, table, count, eased, stretch, rim);
}

// Value and normal from one walk of the spine, as `sweep_value_normal` is and for the measured
// reason recorded there: the driver unrolls that walk into every call site.
fn loft_rim_value_normal(
    spine: u32,
    frame: u32,
    p: vec3<f32>,
    table: u32,
    count: u32,
    eased: f32,
    stretch: f32,
    rim: vec4<f32>,
) -> vec4<f32> {
    let span = curve_spans[spine];
    if span.count < 2u {
        return vec4<f32>(0.0, 0.0, 1.0, 1e30);
    }
    let fspan = curve_spans[frame];
    let best_i = nearest_segment(span, p).i;
    let h = 1e-3;
    var value = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 7u; k = k + 1u) {
        var step = vec3<f32>(0.0, 0.0, 0.0);
        var weight = 0.0;
        if k > 0u {
            let axis = (k - 1u) / 2u;
            weight = 1.0 - 2.0 * f32((k - 1u) % 2u);
            step[axis] = h * weight;
        }
        let reading = loft_on(span, fspan, best_i, p + step, table, count, eased, stretch, rim);
        if k == 0u {
            value = reading;
        } else {
            n[(k - 1u) / 2u] += weight * reading;
        }
    }
    let l = length(n);
    if l < 1e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, value);
    }
    return vec4<f32>(n / l, value);
}

// ---- end of profile solids ----

fn pipe_distance(index: u32, p: vec3<f32>, radius: f32) -> f32 {
    let span = curve_spans[index];
    if span.count == 0u {
        return 1e30;
    }
    if span.count == 1u {
        return distance(p, curve_points[span.start].xyz) - radius;
    }
    return distance(p, nearest_segment(span, p).q) - radius;
}

// Distance from a baked cage, read out of the atlas.
//
// Outside the bake the grid has nothing to say, so the answer is the distance to the bake's own
// box. The cage surface is inside that box, so its true distance is at least that far -- which
// makes the answer a lower bound rather than a guess, and lower bounds are what a renderer
// stepping toward a surface needs. Overstating would step through it.
/// Trilinearly interpolated texel: normal in `xyz`, distance in `w`.
///
/// Both are interpolated together so the shading and the surface cannot end up in slightly
/// different places, which is also why they share a texture rather than sitting in two.
fn sample_baked_raw(index: u32, p: vec3<f32>) -> vec4<f32> {
    let slab = cage_slabs[index];
    let dims = vec3<f32>(slab.dims.xyz);

    // Grid coordinates within this cage's own bake, then trilinear interpolation between the
    // eight surrounding samples. The cage's slab starts at `layer_offset` in the atlas, and
    // clamping to its own `dims` means the padding around a smaller cage is never read.
    let grid = clamp((p - slab.lo.xyz) * slab.inv_cell.xyz, vec3<f32>(0.0), dims - vec3<f32>(1.0));
    let base = floor(grid);
    let f = grid - base;
    let i0 = vec3<i32>(base);
    let hi_i = vec3<i32>(dims) - vec3<i32>(1);
    let i1 = min(i0 + vec3<i32>(1), hi_i);
    let layer = i32(slab.layer_offset);

    let c000 = textureLoad(cage_atlas, vec3<i32>(i0.x, i0.y, i0.z + layer), 0);
    let c100 = textureLoad(cage_atlas, vec3<i32>(i1.x, i0.y, i0.z + layer), 0);
    let c010 = textureLoad(cage_atlas, vec3<i32>(i0.x, i1.y, i0.z + layer), 0);
    let c110 = textureLoad(cage_atlas, vec3<i32>(i1.x, i1.y, i0.z + layer), 0);
    let c001 = textureLoad(cage_atlas, vec3<i32>(i0.x, i0.y, i1.z + layer), 0);
    let c101 = textureLoad(cage_atlas, vec3<i32>(i1.x, i0.y, i1.z + layer), 0);
    let c011 = textureLoad(cage_atlas, vec3<i32>(i0.x, i1.y, i1.z + layer), 0);
    let c111 = textureLoad(cage_atlas, vec3<i32>(i1.x, i1.y, i1.z + layer), 0);

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn sample_baked(index: u32, p: vec3<f32>) -> f32 {
    let slab = cage_slabs[index];
    let dims = vec3<f32>(slab.dims.xyz);
    let hi = slab.lo.xyz + (dims - vec3<f32>(1.0)) / slab.inv_cell.xyz;

    let outside = box_distance(p, slab.lo.xyz, hi);
    if outside > 0.0 {
        // Outside the bake, combine the distance to the box with what the bake says at the
        // point of the box nearest `p`.
        //
        // Two wrong answers were tried first, and the shape of both is worth keeping. Returning
        // `max(outside, band)` treats the band as a floor; it is not a floor but a claim, and a
        // false one, because the surface can sit closer to a face than the band is wide. The
        // shader then promises more clear room than exists and the ray steps through the
        // surface — 189 mm of error on a 1000 mm cage, appearing and disappearing with the
        // viewing angle, since which face a ray enters through is what decides it. Returning
        // `outside` alone is honest but useless: it falls to zero *at* the face, so the hit
        // test fires on the box and the box itself becomes the model. That was worse, 840 mm.
        //
        // The bound that is both valid and tight: `q` is the point of the box nearest `p`, and
        // every surface point lies inside the box, so the angle at `q` between `p - q` and any
        // `s - q` is at least a right angle. By the law of cosines
        // `|p - s|^2 >= |p - q|^2 + |q - s|^2`, and the bake already stores a lower bound on
        // `|q - s|`. It grows away from the box instead of collapsing at it.
        let q = clamp(p, slab.lo.xyz, hi);
        let at_face = max(sample_baked_raw(index, q).w, 0.0);
        return sqrt(outside * outside + at_face * at_face);
    }
    return sample_baked_raw(index, p).w;
}

/// The cage's own normal, carried through the bake rather than recovered from it.
///
/// This is the fix for a traced surface shading as though it were tessellated. Differencing the
/// stored *distance* recovers whatever the sampling made piecewise-constant — the grid cells,
/// and beneath them the triangulation the cage field is built on — and measured 4.56 degrees
/// off the true normal at the viewport's settings, against 1.48 for faceting already judged
/// visible. The stored normal measures 0.03. `bake_normal_accuracy` is where those come from.
///
/// It also means shading no longer depends on the bake resolution: a coarse bake is 0.04
/// degrees, the same as a fine one, because the grid now only has to place the surface.
fn sample_baked_normal(index: u32, p: vec3<f32>) -> vec3<f32> {
    let slab = cage_slabs[index];
    let dims = vec3<f32>(slab.dims.xyz);
    let hi = slab.lo.xyz + (dims - vec3<f32>(1.0)) / slab.inv_cell.xyz;

    // Outside the bake the grid has no normal to give, so point away from its box — the same
    // reasoning that makes the distance there the distance to the box.
    let clamped = clamp(p, slab.lo.xyz, hi);
    if box_distance(p, slab.lo.xyz, hi) > 0.0 {
        return normalize(p - clamped + vec3<f32>(0.0, 0.0, 1e-9));
    }

    let n = sample_baked_raw(index, p).xyz;
    let len = length(n);
    if len < 1e-9 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return n / len;
}

/// Outward normal of a swept tube: away from the nearest point on its spine.
fn pipe_normal(index: u32, p: vec3<f32>) -> vec3<f32> {
    let span = curve_spans[index];
    if span.count == 0u {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let away = p - nearest_segment(span, p).q;
    let len = length(away);
    if len < 1e-9 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return away / len;
}

// **A run of pipes one union takes, walked rather than written out** — item 34's fix one level up.
//
// A union of `n` pipes used to be `n` calls, and the driver inlines each of them at each of the
// six places `march` reads the field, so the source grew by a line an operand and the *compiled*
// function grew by 1 835 — a spine walk and its five loops, six times over. The driver's compile
// is a power of about 2–2.5 in that size, so sixteen pipes cost 123 seconds against two pipes'
// 0.65. Measured in `union-compile-8fa06b7.md`.
//
// The operands differ by two integers. A pipe is a spine index and a radius slot, both of which
// live in buffers already, so the whole list collapses to one walk behind a loop: `first + k` and
// `radius_slot + k`, which is why codegen interns the run's spines consecutively and pushes its
// radii consecutively. **The count is a `params` read, not a literal**, for the two reasons item
// 34 gives: a literal bound is one a driver may unroll, which would undo the change, and a count
// in the buffer means adding a pipe to a run changes no shader text at all.
//
// Nought copies is the answer for a union whose operands are all in the run: `1e30` is what a
// `min` ignores, so an empty run folds away, and every other operand folds against this one
// exactly as it did.
fn pipe_run_distance(first: u32, radius_slot: u32, count: u32, p: vec3<f32>) -> f32 {
    var best = 1e30;
    for (var k = 0u; k < count; k = k + 1u) {
        best = min(best, pipe_distance(first + k, p, params[radius_slot + k]));
    }
    return best;
}

// The same run, carrying the normal of whichever pipe defined the value, in the `vec4` shape a
// body's normal function already packs one: `xyz` the normal, `w` the value.
//
// **Strictly less, so a tie goes to the earlier pipe**, which is the rule `GraphField::normal_at`
// folds a boolean by and the rule `fold_boolean_with_normal` writes out. The loop walks the run in
// the order the operand list holds it, so "earlier" means the same thing on both paths.
fn pipe_run_nearest(first: u32, radius_slot: u32, count: u32, p: vec3<f32>) -> vec4<f32> {
    var best = vec4<f32>(0.0, 0.0, 1.0, 1e30);
    for (var k = 0u; k < count; k = k + 1u) {
        let v = pipe_distance(first + k, p, params[radius_slot + k]);
        if (v < best.w) {
            best = vec4<f32>(pipe_normal(first + k, p), v);
        }
    }
    return best;
}

// **A run of extrudes one union takes, walked rather than written out** -- the pipe run's
// counterpart on the one block that is not a curve walk.
//
// The same fix for a different reason, which is why the pipe run was not enough on its own.
// `extrude_spokes_cost.rs` put sixteen blades on spokes against sixteen pipes: 124 s to 120, so
// the blow-up is not the spine walk. An extrude's operand is 2 328 inlined lines against a pipe's
// 1 835, and most of that is `extrude_normal` -- it central-differences, so it inlines
// `extrude_distance` six times on its own, and the driver then inlines the whole composition six
// times over.
//
// What makes the loop possible is that **every extrude reaches the shader the same way**:
// `extrude_placement` resolves the profile's plane, its rotation and the sweep direction to nine
// parameter slots whichever `Sweep` the node holds, and its depth is six more (`ExtrudeDepth`).
// So the operands differ by a profile binding and fifteen consecutive slots, and the list
// collapses to one walk: `first + k` and `slot + 15 * k`.
//
// **Fifteen is written in two places** -- here and `RunKind::slots_each` -- and they are the same
// number by agreement rather than by construction. The two passes' slot counting is what would
// drift if they parted, and `a_looped_run_pushes_the_same_slots_in_both_passes` is what catches
// it, on extrudes as on pipes.
//
// The count is a `params` read, not a literal, for the reasons `pipe_run_distance` gives: a
// literal bound is one a driver may unroll, which would undo the change, and a count in the
// buffer means adding a blade to a fan changes no shader text at all.
// An extrude's six depth slots, read from where a run's member has them.
fn extrude_depth_at(s: u32) -> ExtrudeDepth {
    return ExtrudeDepth(
        params[s + 9u], params[s + 10u], params[s + 11u], params[s + 12u], params[s + 13u], params[s + 14u]
    );
}

fn extrude_run_distance(first: u32, slot: u32, count: u32, p: vec3<f32>) -> f32 {
    var best = 1e30;
    for (var k = 0u; k < count; k = k + 1u) {
        let s = slot + k * 15u;
        let at = vec3<f32>(params[s], params[s + 1u], params[s + 2u]);
        let euler = vec3<f32>(params[s + 3u], params[s + 4u], params[s + 5u]);
        let dir = vec3<f32>(params[s + 6u], params[s + 7u], params[s + 8u]);
        let v = extrude_distance(first + k, to_local(p, at, euler, 1.0), dir, extrude_depth_at(s));
        best = min(best, v);
    }
    return best;
}

// The same run, carrying the normal of whichever extrude defined the value, in the `vec4` shape a
// body's normal function already packs one: `xyz` the normal, `w` the value.
//
// **Strictly less, so a tie goes to the earlier extrude**, which is the rule
// `GraphField::normal_at` folds a boolean by and the rule `fold_boolean_with_normal` writes out.
// The loop walks the run in the order the operand list holds it, so "earlier" means the same
// thing on both paths.
//
// **The normal is taken inside the branch**, which is where most of the saving is: written out an
// operand, `extrude_normal`'s six difference calls are what the driver inlines at every one of
// `march`'s six field reads. Asked once for the winner, they are one copy.
fn extrude_run_nearest(first: u32, slot: u32, count: u32, p: vec3<f32>) -> vec4<f32> {
    var best = vec4<f32>(0.0, 0.0, 1.0, 1e30);
    for (var k = 0u; k < count; k = k + 1u) {
        let s = slot + k * 15u;
        let at = vec3<f32>(params[s], params[s + 1u], params[s + 2u]);
        let euler = vec3<f32>(params[s + 3u], params[s + 4u], params[s + 5u]);
        let dir = vec3<f32>(params[s + 6u], params[s + 7u], params[s + 8u]);
        let local = to_local(p, at, euler, 1.0);
        let depth = extrude_depth_at(s);
        let v = extrude_distance(first + k, local, dir, depth);
        if (v < best.w) {
            best = vec4<f32>(extrude_normal(first + k, local, dir, depth, euler), v);
        }
    }
    return best;
}

// **A run of thin extrudes one union takes, walked rather than written out** -- the third kind of
// run, and the one that needed it most.
//
// A wall is a heavier operand than a profile. Its call carries the walk of its chain and its
// bare edges beside the lookup a baked profile is, and `wall_extrude_normal` differences all of
// that six times. Written out an operand, a union of thin extrudes took the driver 6 seconds to
// compile at two ribs, 25 at four and 178 at eight, where eight thick blades in their loop took
// 0.3 (`wall_cost.rs`).
//
// The operands differ the way thick extrudes do, by what they bind and what is in their slots, so
// the list collapses to one walk in the same way: a wall's three polylines at `first + 3k` and the
// two after it, and its nineteen slots at `slot + 19k` -- a thick extrude's fifteen, in the
// order `extrude_run_distance` reads them, then the thickness, which way the band lies, whether
// the ends are round and whether the chain closes.
//
// **Three and nineteen are each written in two places** -- here and `RunKind` -- and are the
// same numbers by agreement rather than by construction, as fifteen is.
// `a_fan_of_thin_extrudes_is_one_walk` is what catches the two parting.
//
// The count is a `params` read and not a literal, for the reasons `pipe_run_distance` gives.
fn wall_extrude_run_distance(first: u32, slot: u32, count: u32, p: vec3<f32>) -> f32 {
    var best = 1e30;
    for (var k = 0u; k < count; k = k + 1u) {
        let s = slot + k * 19u;
        let c = first + k * 3u;
        let at = vec3<f32>(params[s], params[s + 1u], params[s + 2u]);
        let euler = vec3<f32>(params[s + 3u], params[s + 4u], params[s + 5u]);
        let dir = vec3<f32>(params[s + 6u], params[s + 7u], params[s + 8u]);
        let v = wall_extrude_distance(
            c,
            c + 1u,
            c + 2u,
            to_local(p, at, euler, 1.0),
            dir,
            extrude_depth_at(s),
            params[s + 15u],
            params[s + 16u],
            params[s + 17u],
            params[s + 18u]
        );
        best = min(best, v);
    }
    return best;
}

// The same run, carrying the normal of whichever wall defined the value, in the `vec4` shape a
// body's normal function already packs one: `xyz` the normal, `w` the value.
//
// **Strictly less, so a tie goes to the earlier member**, and **the normal is taken inside the
// branch** -- both for the reasons `extrude_run_nearest` gives, and the second counts for more
// here: six differenced copies of a wall are what the loop asks for once, for the winner.
fn wall_extrude_run_nearest(first: u32, slot: u32, count: u32, p: vec3<f32>) -> vec4<f32> {
    var best = vec4<f32>(0.0, 0.0, 1.0, 1e30);
    for (var k = 0u; k < count; k = k + 1u) {
        let s = slot + k * 19u;
        let c = first + k * 3u;
        let at = vec3<f32>(params[s], params[s + 1u], params[s + 2u]);
        let euler = vec3<f32>(params[s + 3u], params[s + 4u], params[s + 5u]);
        let dir = vec3<f32>(params[s + 6u], params[s + 7u], params[s + 8u]);
        let local = to_local(p, at, euler, 1.0);
        let depth = extrude_depth_at(s);
        let thickness = params[s + 15u];
        let facing = params[s + 16u];
        let round = params[s + 17u];
        let closed = params[s + 18u];
        let v = wall_extrude_distance(
            c, c + 1u, c + 2u, local, dir, depth, thickness, facing, round, closed
        );
        if (v < best.w) {
            best = vec4<f32>(
                wall_extrude_normal(
                    c, c + 1u, c + 2u, local, dir, depth, thickness, facing, round, closed, euler
                ),
                v
            );
        }
    }
    return best;
}

// A frame's axis as a unit vector.
//
// A `normalize` would do, except for the one case the kernel goes out of its way to allow: an
// axis of nothing at all means `+Z` rather than a division by zero, because that is a state you
// pass through while typing a vector one component at a time, and refusing it would turn typing
// into an error.
//
// **The same length and the same comparison as `rotation_from_z`**, which is
// `try_normalize(AXIS_FLOOR)` and gives up at `n <= 1e-9`, not below it. Written at 1e-12 first,
// which left a band between the two where the field had given up on a direction and the shader
// was still lighting a surface by it. `the_shader_gives_up_on_an_axis_at_the_same_length_the_
// field_does` is what stops the two numbers drifting apart again.
fn aimed(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l <= 1e-9) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v / l;
}

// A point folded onto whichever side of a plane its body is on.
//
// **What a joined mirror writes when its body never crosses its own plane.** The general form is
// `min(f(p), f(mirror(p)))`, which writes the body out twice; folding first asks once, and asks
// the same question, because the query's own half is always the nearer one when the body is all
// on one side. `fieldform_field::mirror_one_side` carries the inequality and the measurement.
//
// `side` is `+1` or `-1`, for the plane's positive or negative side.
fn mirror_fold(p: vec3<f32>, at: vec3<f32>, normal: vec3<f32>, side: f32) -> vec3<f32> {
    let across = dot(p - at, normal);
    return select(p, p - normal * (2.0 * across), across * side < 0.0);
}

// A point reflected in the plane through `at` with unit `normal`.
//
// A general plane rather than an axis, because a mirror's plane can be a `Plane` node as well
// as one of the frame's axes, and both resolve to a point and a direction before they get here.
fn mirror_point(p: vec3<f32>, at: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return p - normal * (2.0 * dot(p - at, normal));
}

// The matching reflection for a direction: minus twice its component along the normal. Still
// unit length afterwards, because a reflection is an isometry.
fn mirror_normal(n: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return n - normal * (2.0 * dot(n, normal));
}

// A point turned back about a twist's axis by its own rate, so the inner field is asked where
// the material came from. The inverse of the warp, which is what evaluating one means.
//
// Done in a frame where the chosen axis is the third coordinate, so one piece of arithmetic
// serves all three. The frames are the cyclic permutations, which are rotations rather than
// reflections — a swap would turn the body inside out on two of the three axes.
// What an attracted body's value is divided by to stay a lower bound on the true distance.
//
// `fieldform_field::attract::stretch`, and asymmetric for the reason stated there: a push does
// not stretch by the same factor a pull of the same size does, and treating it as if it did
// leaves the field claiming room it has not got.
fn attract_stretch(strength: f32) -> f32 {
    if strength >= 0.0 {
        return 1.0 + strength;
    }
    return 1.0 - strength * (11.0 / 16.0);
}

// Where to sample the unwarped field, for a body drawn toward a curve.
//
// The other half of `fieldform_field::attract::warp_point`, and the two have to agree to the
// last bit for the same reason every other pair here does: `gpu_matches_cpu` compares the
// pictures, and a falloff that differed only in the third decimal would show as a soft edge in
// one and a hard one in the other. The constants are not repeated — the shape of the
// polynomial is, and it is the whole of what there is to repeat.
//
// Pushes the point **away**, which draws the material toward. See the module note on the CPU
// side; this is the place the sign is easiest to get backwards and hardest to notice.
// Where a curve was met: the nearest point, and which way the curve runs there.
//
// `along` is the zero vector at a joint or an end, where the curve has no one direction. That
// is a real distinction rather than a missing value -- see `Nearest::along` on the CPU side.
struct CurveHit {
    point: vec3<f32>,
    along: vec3<f32>,
}

// The nearest point on a curve's spine to `p`, and the curve's direction there.
//
// Split out because two things need it and they must agree: the warp moves the sample point
// away from this point, and the normal is turned using the same point and direction. A normal
// derived from a different search than the warp used is a surface lit as though it faced
// somewhere it does not, along exactly the seam where two segments meet.
//
// The nearest *point* rather than the nearest distance, which is why this is not
// `pipe_distance` with one line changed: that one stays in squared distances and never has to
// say where. The CPU keeps the same two searches apart, for the same reason.
fn closest_on_curve(index: u32, p: vec3<f32>) -> CurveHit {
    let span = curve_spans[index];
    var hit: CurveHit;
    hit.point = curve_points[span.start].xyz;
    hit.along = vec3<f32>(0.0);
    if span.count > 1u {
        let near = nearest_segment(span, p);
        hit.point = near.q;
        let ab = curve_points[span.start + near.i + 1u].xyz - curve_points[span.start + near.i].xyz;
        let denom = dot(ab, ab);
        // Only where the segment was met along its length. At either end the nearest point
        // is a joint or a cap and the curve has no one direction there.
        if near.t > 1e-6 && near.t < 1.0 - 1e-6 && denom > 1e-18 {
            hit.along = ab * inverseSqrt(denom);
        }
    }
    return hit;
}

fn attract_point(index: u32, p: vec3<f32>, strength: f32, falloff: f32) -> vec3<f32> {
    let span = curve_spans[index];
    if span.count == 0u || falloff <= 0.0 || strength == 0.0 {
        return p;
    }
    let away = p - closest_on_curve(index, p).point;
    let x = clamp(length(away) / falloff, 0.0, 1.0);
    return p + away * (strength * (1.0 - x * x * (3.0 - 2.0 * x)));
}

// The normal of a body drawn toward a curve, from the unwarped body's normal at the warped
// point.
//
// **A warp moves a surface and turns it, and those are not the same transformation.** The field
// is `F(g(p))`, so its gradient is `Dg^T` times the inner gradient, and for this warp `Dg` is
// symmetric with three distinct factors: `phi'(d)` out from the curve, `1 + w` around it, and
// **1** along it. The third is the one that is easy to miss -- slide `p` parallel to a straight
// run of curve and the nearest point slides with it, so nothing is scaled at all.
//
// `fieldform_field::attract::warp_normal` is the same arithmetic, and the two must agree: this
// is what lights the surface on screen, and that is what a mesh or a measurement would use.
// `an_attracted_surface_faces_where_its_own_field_says_it_does` holds the CPU side against the
// field's own gradient; nothing holds this one except being the same six lines.
fn attract_normal(
    index: u32,
    p: vec3<f32>,
    strength: f32,
    falloff: f32,
    inner: vec3<f32>,
) -> vec3<f32> {
    let span = curve_spans[index];
    if span.count == 0u || falloff <= 0.0 || strength == 0.0 {
        return inner;
    }
    let hit = closest_on_curve(index, p);
    let away = p - hit.point;
    let d = length(away);
    // On the curve there is no radial direction, and beyond the reach the warp is the identity.
    // Neither is a failure: both are places the inner normal is already the right answer.
    if d < 1e-9 || d >= falloff {
        return inner;
    }

    let x = d / falloff;
    let across = 1.0 + strength * (1.0 - x * x * (3.0 - 2.0 * x));
    let radial = 1.0 + strength * (1.0 - 9.0 * x * x + 8.0 * x * x * x);

    let n_hat = away / d;
    var turned = inner * across + n_hat * (dot(inner, n_hat) * (radial - across));
    // The axial term, which undoes the `across` factor along the curve. `along` is zero at a
    // joint, where there is no axis to undo it along and the geometry is a sphere instead.
    let axial = dot(hit.along, hit.along);
    if axial > 0.5 {
        turned = turned + hit.along * (dot(inner, hit.along) * (1.0 - across));
    }
    // At the fold the radial factor passes through zero, and a normal that was purely radial
    // there has nothing left to normalize. The inner normal is the honest fallback: it is
    // wrong, but it is a direction, and the alternative is a NaN that paints the pixel black --
    // which is the very artefact this is here to remove.
    let length_of = length(turned);
    if length_of < 1e-12 {
        return inner;
    }
    return turned / length_of;
}

fn unwind_point(p: vec3<f32>, radians_per_mm: f32, axis: f32, at: vec3<f32>) -> vec3<f32> {
    let d = p - at;
    var local: vec3<f32>;
    if axis < 0.5 {
        local = vec3<f32>(d.y, d.z, d.x);
    } else if axis < 1.5 {
        local = vec3<f32>(d.z, d.x, d.y);
    } else {
        local = d;
    }
    let angle = -radians_per_mm * local.z;
    let c = cos(angle);
    let s = sin(angle);
    let turned = vec3<f32>(local.x * c - local.y * s, local.x * s + local.y * c, local.z);
    var world: vec3<f32>;
    if axis < 0.5 {
        world = vec3<f32>(turned.z, turned.x, turned.y);
    } else if axis < 1.5 {
        world = vec3<f32>(turned.y, turned.z, turned.x);
    } else {
        world = turned;
    }
    return world + at;
}

/// Where a point sits relative to one instance of a linear pattern.
///
/// **The fold, which is what stops a pattern costing its own count.** A pattern's instances are a
/// regular lattice of one body, so the instance nearest a point is the one whose cell the point
/// falls in — found by arithmetic rather than by trying them all, exactly as a lattice finds its
/// cell. The caller asks for a small window of cells around that one, because a body that spills
/// out of its own cell can be nearest from a neighbouring one; `pattern_reach` in the field
/// crate decides how wide, and both sides ask it rather than each deciding.
///
/// `offset` walks that window. A row stops at its ends — a point past the last copy belongs to
/// the last copy — which is what the clamp is.
///
/// `origin` is where the body itself sits along the step, and the cell is counted from there:
/// instance `i` is the body moved `i` steps, so the instance covering a point is the number of
/// steps between the body and the point. `pattern_origin` in the field crate is what supplies
/// it, to both sides, so the two folds cannot centre their windows in different places.
/// Returns the step's point in `xyz` and **which instance it belongs to** in `w`, because the
/// caller has to be able to leave one out: a block that does not hold instance zero — the body
/// where it already is — must not fold it in. See `Node::Pattern::join`.
// The least a pattern's sub-field can read at `q`, from the sphere its shape lies in: five
// slots from `slot` — the sphere's centre and radius in the sub-field's own frame, and the
// largest divisor anything in the sub-field applies to its value.
//
// **A floor, and only where it is one.** Every leaf the generator admits here reads at least
// the distance to a sphere that holds it, so outside that sphere the sub-field reads at least
// `(|q − c| − ρ) / D`, and an instance whose floor is no less than the running minimum cannot
// lower it. Inside the sphere nothing is promised — a point deep in the body reads far below
// any such figure — so the answer there is "evaluate it". See `pattern_prune_floor` in
// codegen.rs for which sub-fields are admitted and why the rest are not.
// ---- Which copies of a pattern matter here (plan section 5, V6; `COPY_MASK` in codegen.rs) ----
//
// Module variables for the reason `owner` and `crit_gap` are: what they carry is a fact about
// the point `field` was asked about, and the pattern that reads it may sit any depth below.
//
// `mask_off` is the copies to pass over, one bit each. **Nought unless somebody has said
// otherwise**, which is what a private variable starts as and is the safe reading: every
// caller of `field` that knows nothing about masks -- the normal, the lattice of points, the
// skip grid's own fill -- walks every copy, as it always did. Only `field_here` in march.wgsl
// sets it, for the one call it makes, and puts it back.
var<private> mask_off: u32;
// While the skip grid is being filled: `mask_mode` is one, `mask_slack` is twice the cell's
// half diagonal, and the pattern answers in `mask_kept` with the copies the cell keeps.
// `mask_seen` says the pattern was reached at all -- a body's box guard may have passed it
// over at the cell's centre, and a pattern that was never asked has ruled nothing out.
var<private> mask_mode: u32;
var<private> mask_slack: f32;
var<private> mask_kept: u32;
var<private> mask_seen: bool;

// The same three, for the operands of a join (`OperandMasks` in codegen.rs): the ones to pass
// over here, and while the grid is being filled the ones this cell keeps and the ones that
// were asked at all. One bit an operand, numbered by the generator.
var<private> ops_off: u32;
var<private> ops_kept: u32;
var<private> ops_seen: u32;

fn operand_masked(bit: u32) -> bool {
    return ((ops_off >> bit) & 1u) == 1u;
}

fn copy_masked(index: i32) -> bool {
    // A shift by thirty-two or more is not defined, and a copy past the mask is never skipped.
    if index < 0 || index >= 32 {
        return false;
    }
    return ((mask_off >> u32(index)) & 1u) == 1u;
}

// ---- A grid: rows and columns of copies --------------------------------------------------------
//
// Copy `index` is column `index mod n1` of row `index div n1`, and stands `disp0 + a*e1 + b*e2` from
// the body. The window around a point is centred on the copy nearest it *among those that exist*,
// found as `Fold::grid_window` finds it in fieldform-field: the minimiser of the distance over the
// box of copies, which for a convex quadratic is the unconstrained solution when it is inside and
// otherwise the best of the four edges and four corners. Clamping the two coordinates one at a time
// is right only for a square grid -- at an oblique one the nearest copy to a point past an edge is
// found by sliding along the edge to where the point projects.

// Offers `(a, b)`, clamped to the box, as the nearest so far. `best` is the squared distance and
// the column and row it was found at.
fn pattern_grid_offer(
    u: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    best: vec3<f32>,
    a: f32,
    b: f32,
    hi: vec2<f32>,
) -> vec3<f32> {
    let ca = clamp(a, 0.0, hi.x);
    let cb = clamp(b, 0.0, hi.y);
    let r = u - e1 * ca - e2 * cb;
    let cost = dot(r, r);
    if cost < best.x {
        return vec3<f32>(cost, ca, cb);
    }
    return best;
}

// The (column, row) of the copy nearest `p`, as numbers: the centre of the window.
fn pattern_grid_centre(
    p: vec3<f32>,
    anchor: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    d1: vec3<f32>,
    d2: vec3<f32>,
    n1: f32,
    n2: f32,
) -> vec2<f32> {
    let u = p - anchor;
    let hi = vec2<f32>(n1 - 1.0, n2 - 1.0);
    let e1e1 = dot(e1, e1);
    let e2e2 = dot(e2, e2);
    var best = vec3<f32>(3.4e38, 0.0, 0.0);
    best = pattern_grid_offer(u, e1, e2, best, dot(u, d1), dot(u, d2), hi);
    best = pattern_grid_offer(u, e1, e2, best, 0.0, dot(u, e2) / e2e2, hi);
    best = pattern_grid_offer(u, e1, e2, best, hi.x, dot(u - e1 * hi.x, e2) / e2e2, hi);
    best = pattern_grid_offer(u, e1, e2, best, dot(u, e1) / e1e1, 0.0, hi);
    best = pattern_grid_offer(u, e1, e2, best, dot(u - e2 * hi.y, e1) / e1e1, hi.y, hi);
    best = pattern_grid_offer(u, e1, e2, best, 0.0, 0.0, hi);
    best = pattern_grid_offer(u, e1, e2, best, hi.x, 0.0, hi);
    best = pattern_grid_offer(u, e1, e2, best, 0.0, hi.y, hi);
    best = pattern_grid_offer(u, e1, e2, best, hi.x, hi.y, hi);
    return best.yz;
}

// Copy `index` applied to a point: carried back by the motion that put the copy where it stands.
fn pattern_slide_grid(
    p: vec3<f32>,
    disp0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    n1: f32,
    index: f32,
) -> vec3<f32> {
    let b = floor(index / n1);
    let a = index - b * n1;
    return p - (disp0 + e1 * a + e2 * b);
}

// The window's copy `(centre + da, centre + db)`: its point, and its number in `w`, or -1 where
// it is off the grid and there is no such copy.
fn pattern_point_grid(
    p: vec3<f32>,
    disp0: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    centre: vec2<f32>,
    n1: f32,
    n2: f32,
    da: f32,
    db: f32,
) -> vec4<f32> {
    let a = round(centre.x) + da;
    let b = round(centre.y) + db;
    if (a < 0.0 || a >= n1 || b < 0.0 || b >= n2) {
        return vec4<f32>(p, -1.0);
    }
    return vec4<f32>(p - (disp0 + e1 * a + e2 * b), a + n1 * b);
}

// Whether a pattern leaves copy `index` out, read from the words its skip list is stored in from
// `slot`: copy `i` is bit `i % 24` of word `i / 24`. Twenty-four, because a parameter is an `f32`
// and holds an integer exactly only to 2^24. The caller has already seen that `index` is a copy.
fn pattern_skipped(slot: u32, index: i32) -> bool {
    let word = u32(params[slot + u32(index / 24)]);
    return ((word >> u32(index % 24)) & 1u) == 1u;
}

fn pattern_floor(q: vec3<f32>, slot: u32) -> f32 {
    let c = vec3<f32>(params[slot], params[slot + 1u], params[slot + 2u]);
    let outside = length(q - c) - params[slot + 3u];
    if outside <= 0.0 {
        return -1e30;
    }
    return outside / params[slot + 4u];
}

// The copy of a ring a point's angle belongs to, in `0 ..< count`, matching `Fold::cell` in
// fieldform-field. `origin` is the angle of copy 0's body and `step` the angle between neighbours.
//
// A full turn wraps: the cell is taken modulo the count. An arc does not, and a point behind it
// belongs to whichever end is nearer. The point's offset from copy 0 is taken the short way round
// *about the arc's middle*, so the two ends swap exactly at the arc's far side -- which is where
// the nearer one changes -- and clamping the cell then lands on the end that is nearest.
fn pattern_ring_cell(angle: f32, count: f32, step: f32, origin: f32, arc: f32) -> f32 {
    if (arc > 0.5) {
        let middle = (count - 1.0) * 0.5 * step;
        var offset = angle - origin;
        offset = offset - 6.2831853071795864 * round((offset - middle) / 6.2831853071795864);
        return clamp(round(offset / step), 0.0, count - 1.0);
    }
    let cell = round((angle - origin) / step);
    return cell - floor(cell / count) * count;
}

// The instance a point belongs to, by the same arithmetic the windowed fold centres itself
// with, so a loop that walks every instance can start at the nearest and work outward.
fn pattern_nearest_radial(
    p: vec3<f32>,
    at: vec3<f32>,
    euler: vec3<f32>,
    count: f32,
    step: f32,
    origin: f32,
    arc: f32,
) -> f32 {
    let m = euler_rotation(euler);
    let local = (p - at) * m;
    return pattern_ring_cell(atan2(local.y, local.x), count, step, origin, arc);
}

fn pattern_nearest_linear(p: vec3<f32>, unit: vec3<f32>, pitch: f32, count: f32, origin: f32) -> f32 {
    if (pitch <= 1e-6) {
        return 0.0;
    }
    return clamp(round((dot(p, unit) - origin) / pitch), 0.0, count - 1.0);
}

fn pattern_point_linear(
    p: vec3<f32>,
    unit: vec3<f32>,
    pitch: f32,
    count: f32,
    origin: f32,
    bias: f32,
    offset: f32,
) -> vec4<f32> {
    if (pitch <= 1e-6) {
        return vec4<f32>(p, 0.0);
    }
    // **The centre is clamped and the window is not**, which is not the same thing as clamping
    // the result — and the difference only shows once the fold does something other than take a
    // minimum. Clamping each step piled several steps onto the end instance: harmless for a
    // `min`, which is idempotent, and wrong for the sum a rounded fold accumulates, where that
    // instance was then counted three or four times and pushed the surface out by half a
    // radius. The caller drops an index outside the row; this only keeps the *centre* of the
    // window on a copy that exists, which is what "a point past the last copy belongs to the
    // last copy" means.
    //
    // `origin` already includes the steps from the body to copy 0 (`bias`, nought unless the
    // copies are centred on the body), so the cell counted from it is the copy's own number; the
    // copy then stands `cell + bias` steps from the body, which is what it is slid by.
    let centre = clamp(round((dot(p, unit) - origin) / pitch), 0.0, count - 1.0);
    let cell = centre + offset;
    return vec4<f32>(pattern_slide_linear(p, unit, pitch, cell + bias), cell);
}

/// Instance `index` of a row, applied to a point: the inverse of stepping it that many times.
fn pattern_slide_linear(p: vec3<f32>, unit: vec3<f32>, pitch: f32, index: f32) -> vec3<f32> {
    return p - unit * (pitch * index);
}

/// Instance `index` of a ring, applied to a point: turned back by `index + bias` steps about the
/// frame's axis.
///
/// Shared with [`pattern_point_radial`], which is this with the index worked out from the
/// point's own angle rather than handed in. A fold that has to visit every instance walks the
/// indices directly, and it must turn the point by the same rotation the windowed form does.
/// `step` is the angle between neighbours, which is a full turn over the count unless the ring
/// stops short, and `bias` the steps from the body to copy 0 (nought unless centred).
fn pattern_spin_radial(
    p: vec3<f32>,
    at: vec3<f32>,
    euler: vec3<f32>,
    step: f32,
    bias: f32,
    index: f32,
) -> vec3<f32> {
    let m = euler_rotation(euler);
    let local = (p - at) * m;
    let turn = -step * (index + bias);
    let c = cos(turn);
    let s = sin(turn);
    let spun = vec3<f32>(local.x * c - local.y * s, local.x * s + local.y * c, local.z);
    return m * spun + at;
}

/// The direction half of the same: a normal found in instance `index`, turned back out.
fn pattern_spin_normal_radial(
    n: vec3<f32>,
    euler: vec3<f32>,
    step: f32,
    bias: f32,
    index: f32,
) -> vec3<f32> {
    let m = euler_rotation(euler);
    let turn = step * (index + bias);
    let c = cos(turn);
    let sn = sin(turn);
    let inner = n * m;
    let spun = vec3<f32>(inner.x * c - inner.y * sn, inner.x * sn + inner.y * c, inner.z);
    return m * spun;
}

/// The same for a turn, where the window wraps rather than stopping.
///
/// There is no end to be past on a full turn, so the cell is taken modulo the count. Written as
/// a rotation of the point *back* by the instance's angle, which is the inverse placement the
/// field stores — the two are the same statement and have to stay that way.
/// The same for a turn, and the instance index comes back in `w` for the same reason.
fn pattern_point_radial(
    p: vec3<f32>,
    at: vec3<f32>,
    euler: vec3<f32>,
    count: f32,
    step: f32,
    origin: f32,
    bias: f32,
    arc: f32,
    offset: f32,
) -> vec4<f32> {
    // **The frame's rotation built once, not twice.** Going in through `to_local` and back out
    // through `from_local` builds the same matrix from the same three angles each way -- six
    // trig calls apiece -- and it is the same matrix. A linear fold is a dot product and a
    // subtraction with no transcendentals at all, which is why a row used to tumble noticeably
    // faster than a ring of the same body.
    let m = euler_rotation(euler);
    // `v * m` is `transpose(m) * v`, which for a rotation is its inverse -- the same identity
    // `to_local` relies on, spelled here so the matrix can be shared.
    let local = (p - at) * m;
    // Measured from the body's own angle about the axis, which is `origin`. Counting from zero
    // instead assumes the body straddles the frame's zero angle; a body 125 mm off an axis
    // sits twelve cells from that assumption, and a window four cells wide then never holds
    // the instance the point belongs to.
    //
    // A full turn wraps the window round; an arc clamps the point's own cell to the copies there
    // are and lets the window run off either end, where the caller drops what is not a copy.
    var cell: f32;
    if (arc > 0.5) {
        cell = pattern_ring_cell(atan2(local.y, local.x), count, step, origin, arc) + offset;
    } else {
        cell = round((atan2(local.y, local.x) - origin) / step) + offset;
        cell = cell - floor(cell / count) * count;
    }
    let turn = -step * (cell + bias);
    let c = cos(turn);
    let s = sin(turn);
    let spun = vec3<f32>(local.x * c - local.y * s, local.x * s + local.y * c, local.z);
    return vec4<f32>(m * spun + at, cell);
}


/// A normal found in one instance of a radial pattern, turned back out into world space.
///
/// A linear pattern needs no such thing: its instances differ by a translation, and a
/// translation leaves directions alone. A turn does not, so the normal has to come back by the
/// same angle the point went in by — and it has to be *the same* angle, which is why this
/// recomputes the cell from the same point rather than being handed one.
fn pattern_normal_radial(
    n: vec3<f32>,
    p: vec3<f32>,
    at: vec3<f32>,
    euler: vec3<f32>,
    count: f32,
    step: f32,
    origin: f32,
    bias: f32,
    arc: f32,
    offset: f32,
) -> vec3<f32> {
    // The same matrix serves the point going in and the direction coming out.
    let m = euler_rotation(euler);
    let local = (p - at) * m;
    var cell: f32;
    if (arc > 0.5) {
        cell = pattern_ring_cell(atan2(local.y, local.x), count, step, origin, arc) + offset;
    } else {
        cell = round((atan2(local.y, local.x) - origin) / step) + offset;
        cell = cell - floor(cell / count) * count;
    }
    // Forward by the angle the point was taken back by. Directions have no origin, so the
    // frame's translation plays no part.
    let turn = step * (cell + bias);
    let c = cos(turn);
    let sn = sin(turn);
    let inner = n * m;
    let spun = vec3<f32>(inner.x * c - inner.y * sn, inner.x * sn + inner.y * c, inner.z);
    return m * spun;
}
