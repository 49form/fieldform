// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
// The lighting pass: from the march's normal, depth and owner to a displayed colour.
//
// Deferred, as `shading-spec.md` §3 asks: the geometry pass wrote a lean G-buffer — a normal,
// a depth and a body index per pixel — and this one full-screen pass turns it into light. It
// knows nothing about the field. Whatever traces the surface, this shades it, and every line
// here survives a change to what does the tracing.
//
// # What it evaluates
//
// The glTF metallic-roughness model — Cook-Torrance with a GGX distribution, height-correlated
// Smith visibility and Schlick Fresnel over a Lambert diffuse — under image-based lighting
// (the split-sum approximation: a roughness-prefiltered environment for the specular reflection,
// an irradiance map for the diffuse, and the BRDF's own integral tabulated) plus three analytic
// lights for the highlights a studio dome alone will not place. On top, for car paint, a clear
// coat with its own roughness and Fresnel over an attenuated base whose flakes are a normal map,
// and for brushed metal an anisotropic distribution stretched along the brush direction.
//
// # Where the forms come from
//
// The equations follow Google Filament's (https://google.github.io/filament/Filament.html),
// Apache-2.0, ported from the published forms rather than by copying its code — with two
// exceptions carried over as written, and credited here: the `f0_clear_coat_to_surface`
// polynomial, and the clear-coat layering of `evaluateClearCoatIBL` / `clearCoatLobe`. Filament
// was chosen because it is a proven, documented, shipping model whose car paint is the look the
// owner judges this viewport by; the first version of the coat here had its own arithmetic, and
// got the coat's Fresnel in twice.
//
// # Colour
//
// Everything is linear until the very end. The result is accumulated in a linear buffer,
// tonemapped with AgX — which keeps saturated paint from going to a flat wash where other
// curves do — and only then encoded for display and written to the 8-bit texture egui shows.

struct Shading {
    // rgb: base colour, linear light.  w: metallic.
    base: vec4<f32>,
    // x: roughness  y: clearcoat  z: clearcoat roughness  w: anisotropy
    surface: vec4<f32>,
    // x: flake  y: exposure  z: environment intensity  w: unused
    look: vec4<f32>,
    // Directional lights: xyz is the unit direction *toward* the light, w its intensity. The
    // first is the key, and the only one the march shadowed.
    light_dirs: array<vec4<f32>, 3>,
    light_colors: array<vec4<f32>, 3>,
    // rgb: what a miss shows, linear.
    background: vec4<f32>,
    // rgb: the selection outline, display-encoded — it is interface, not light, and is drawn
    // over the tonemapped image so exposure never dims it. w: unused.
    outline: vec4<f32>,
    // x: the selected body, or 0xffffffff.  y: frames already accumulated into `hdr`.
    // z: the outline's radius in texels of this frame.
    frame: vec4<u32>,
    // The draft read: xyz the unit pull direction, w 1 when it is on.
    draft_dir: vec4<f32>,
    // x: the draft a face needs, degrees  y: the blur either side of it, degrees  z: opacity
    draft_params: vec4<f32>,
    // Display-encoded colours for a face with enough draft and one without.
    draft_pass: vec4<f32>,
    draft_fail: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> shading: Shading;
// The march's output: xyz normal, w depth (negative where nothing was hit).
@group(0) @binding(2) var<storage, read> gbuffer: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> owners: array<u32>;
// Occlusion and shadow, packed as two halves. See `march.wgsl`. Measured, on a still at one in
// four, at the pixels `occlusion_pixels` names and at those that cannot be filled in from them;
// the rest hold `NO_READING`, and are filled in here -- see `measured_light`.
@group(0) @binding(4) var<storage, read> extras: array<u32>;
// The linear image, averaged over the frames of a still, with the selection outline's coverage
// in its alpha. A buffer rather than a texture because it is read and written in one pass,
// which a 16-bit float storage texture cannot be without an optional feature.
@group(0) @binding(5) var<storage, read_write> hdr: array<vec4<f32>>;
// What is shown: display-encoded, 8 bits a channel, in a plain (not sRGB) texture. egui passes
// texture samples through to the screen as they are, so the encoding has to happen here and
// the texture must not be one the sampler would decode — see `FrameTargets` in `shading.rs`.
@group(0) @binding(6) var present: texture_storage_2d<rgba8unorm, write>;

// The environment, prepared on the CPU — see `environment.rs`.
@group(1) @binding(0) var specular_map: texture_cube<f32>;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
// x: scale on F0, y: bias, indexed by (n·v, roughness).
@group(1) @binding(2) var brdf_lut: texture_2d<f32>;
@group(1) @binding(3) var env_sampler: sampler;
// The flake normal map and its repeating sampler — see `flake.rs`.
@group(1) @binding(4) var flake_map: texture_2d<f32>;
@group(1) @binding(5) var flake_sampler: sampler;

const PI: f32 = 3.14159265359;
// The roughness a mirror is clamped up to for the lamps: below this the GGX lobe is narrower
// than a pixel and every highlight becomes a single flickering texel.
const MIN_ROUGHNESS: f32 = 0.045;
// The fraction of the prefiltered chain's top level a perceptual roughness of 1 maps to; set
// by the number of levels the environment was baked with.
const SPECULAR_LEVELS: f32 = 6.0;
// The flake map's world size and encoding, matching `flake.rs`.
const FLAKE_TILE_MM: f32 = 16.0;
const FLAKE_TILT_RANGE: f32 = 0.5;

fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn v_smith_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ggx_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(ggx_v + ggx_l, 1e-7);
}

fn f_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let f = pow(1.0 - v_dot_h, 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * f;
}

fn f_schlick_scalar(f0: f32, v_dot_h: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

// Anisotropic GGX (Burley 2012), with the lobe's two widths along the tangent and bitangent.
fn d_ggx_anisotropic(at: f32, ab: f32, t_dot_h: f32, b_dot_h: f32, n_dot_h: f32) -> f32 {
    let a2 = at * ab;
    let d = vec3<f32>(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    let d2 = dot(d, d);
    let b2 = a2 / max(d2, 1e-7);
    return a2 * b2 * b2 / PI;
}

fn v_smith_anisotropic(
    at: f32, ab: f32,
    t_dot_v: f32, b_dot_v: f32, t_dot_l: f32, b_dot_l: f32,
    n_dot_v: f32, n_dot_l: f32,
) -> f32 {
    let lambda_v = n_dot_l * length(vec3<f32>(at * t_dot_v, ab * b_dot_v, n_dot_v));
    let lambda_l = n_dot_v * length(vec3<f32>(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return 0.5 / max(lambda_v + lambda_l, 1e-7);
}

// Kelemen's visibility for the clear coat: a thin, smooth layer where the full Smith term is
// not worth its cost. Filament's `visibilityClearCoat`.
fn v_kelemen(l_dot_h: f32) -> f32 {
    return 0.25 / max(l_dot_h * l_dot_h, 1e-4);
}

// The base layer's reflectance once it sits under a coat of IOR 1.5 rather than under air —
// Filament's `f0ClearCoatToSurface`, carried over as written.
fn f0_clear_coat_to_surface(f0: vec3<f32>) -> vec3<f32> {
    return clamp(
        f0 * (f0 * (0.941892 - 0.263008 * f0) + 0.346479) - 0.0285998,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// The base layer's normal under a flaked paint: the flake normal map, as Filament and other renderers
// lay one on the base of a car paint while the coat keeps the geometric normal.
//
// Sampled on the three world planes and blended by the normal, since a field carries no UVs to
// lay a texture along. The level is chosen from the pixel's footprint, so the map's own mips do
// the fading with distance — averaged over a pixel the tilts cancel — and distant paint is a
// smooth sheen while near paint sparkles, with no distance logic here.
fn flaked_normal(n: vec3<f32>, p: vec3<f32>, footprint: f32, strength: f32) -> vec3<f32> {
    let size = f32(textureDimensions(flake_map, 0).x);
    let texel = FLAKE_TILE_MM / size;
    let level = log2(max(footprint / texel, 1.0));
    let scale = 1.0 / FLAKE_TILE_MM;
    var w = pow(abs(n), vec3<f32>(4.0));
    w = w / max(w.x + w.y + w.z, 1e-6);
    // Each plane's map gives a tilt along that plane's own two axes.
    let ox = (textureSampleLevel(flake_map, flake_sampler, p.yz * scale, level).xy * 2.0 - 1.0)
        * FLAKE_TILT_RANGE;
    let oy = (textureSampleLevel(flake_map, flake_sampler, p.zx * scale, level).xy * 2.0 - 1.0)
        * FLAKE_TILT_RANGE;
    let oz = (textureSampleLevel(flake_map, flake_sampler, p.xy * scale, level).xy * 2.0 - 1.0)
        * FLAKE_TILT_RANGE;
    let tilt = w.x * vec3<f32>(0.0, ox.x, ox.y)
        + w.y * vec3<f32>(oy.y, 0.0, oy.x)
        + w.z * vec3<f32>(oz.x, oz.y, 0.0);
    // Only the part across the normal turns it; the rest would stretch it.
    let across = tilt - n * dot(tilt, n);
    return normalize(n + across * strength);
}

// The direction machining marks run in at `n`: world Z projected onto the surface, since a
// turned or milled part's marks follow its axis and a field carries no UVs to say otherwise.
fn brush_tangent(n: vec3<f32>) -> vec3<f32> {
    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if abs(n.z) > 0.99 {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    }
    return normalize(reference - n * dot(reference, n));
}

fn sample_specular(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    let level = roughness * (SPECULAR_LEVELS - 1.0);
    return textureSampleLevel(specular_map, env_sampler, direction, level).rgb;
}

fn sample_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(brdf_lut, env_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
}

struct Surface {
    p: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    open: f32,
    lit: f32,
    footprint: f32,
}

fn shade_surface(s: Surface) -> vec3<f32> {
    let base = shading.base.rgb;
    let metallic = shading.base.w;
    let clearcoat = shading.surface.y;
    let anisotropy = shading.surface.w;
    let flake = shading.look.x;
    let env_intensity = shading.look.z;

    // The coat's roughness as typed for the environment lookup, floored for the lamps.
    let ibl_clearcoat_roughness = clamp(shading.surface.z, 0.0, 1.0);
    let clearcoat_roughness = max(ibl_clearcoat_roughness, MIN_ROUGHNESS);

    // Under a coat the base is at least as rough as the coat: light reaching it has already been
    // scattered once on the way in. Filament's own comment calls this "a hack that will do".
    var perceptual = clamp(shading.surface.x, 0.0, 1.0);
    if clearcoat > 0.0 {
        perceptual = mix(perceptual, max(perceptual, ibl_clearcoat_roughness), clearcoat);
    }
    let ibl_roughness = perceptual;
    let roughness = max(perceptual, MIN_ROUGHNESS);
    let alpha = roughness * roughness;

    var f0 = mix(vec3<f32>(0.04), base, metallic);
    if clearcoat > 0.0 {
        // The base now meets lacquer rather than air, which lowers its reflectance.
        f0 = mix(f0, f0_clear_coat_to_surface(f0), clearcoat);
    }
    let diffuse_color = base * (1.0 - metallic);

    // The geometric normal shades the clearcoat; the base sees the flakes. Nothing about the
    // coat is perturbed — that is the whole look of the material, a smooth lacquer over a
    // glittering base.
    let n_coat = s.n;
    var n = s.n;
    if flake > 0.0 {
        n = flaked_normal(s.n, s.p, s.footprint, flake);
    }
    let v = s.v;
    let n_dot_v = max(dot(n, v), 1e-4);

    // Anisotropy: the highlight stretches along the brush, and the environment reflection
    // bends with it (Filament's bent-normal approximation).
    let tangent = brush_tangent(n);
    let bitangent = cross(n, tangent);
    let at = max(alpha * (1.0 + anisotropy), 1e-3);
    let ab = max(alpha * (1.0 - anisotropy), 1e-3);
    var reflected = reflect(-v, n);
    if abs(anisotropy) > 0.0 {
        var along = tangent;
        if anisotropy >= 0.0 {
            along = bitangent;
        }
        let aniso_tangent = cross(along, v);
        let aniso_normal = cross(aniso_tangent, along);
        let bent = normalize(mix(n, aniso_normal, abs(anisotropy)));
        reflected = reflect(-v, bent);
    }

    // Split-sum image-based lighting. `e` is the lobe's directional albedo, which the diffuse
    // gives up so the two sum to at most one; the energy the single-scatter table loses to
    // multiple bounces is put back for rough metals (Filament's compensation term), without
    // which a rough aluminium darkens toward grey as its roughness rises.
    let dfg = sample_brdf(n_dot_v, ibl_roughness);
    let e = f0 * dfg.x + vec3<f32>(dfg.y);
    let energy_compensation = 1.0 + f0 * (1.0 / max(dfg.x + dfg.y, 1e-4) - 1.0);
    let irradiance = textureSampleLevel(irradiance_map, env_sampler, n, 0.0).rgb;
    let prefiltered = sample_specular(reflected, ibl_roughness);
    let specular_ibl = prefiltered * e * energy_compensation;
    let diffuse_ibl = irradiance * diffuse_color * (vec3<f32>(1.0) - e);
    var color = (diffuse_ibl + specular_ibl) * (env_intensity * s.open);

    // The analytic lights. Only the key was shadow-marched; the fill and rim are the soft
    // wrap-around light a studio has, and a hard shadow from either would read as a second sun.
    for (var i = 0u; i < 3u; i = i + 1u) {
        let l = shading.light_dirs[i].xyz;
        let radiance = shading.light_colors[i].rgb * shading.light_dirs[i].w;
        let n_dot_l = dot(n, l);
        if n_dot_l <= 0.0 {
            continue;
        }
        let h = normalize(v + l);
        let n_dot_h = max(dot(n, h), 0.0);
        let l_dot_h = max(dot(l, h), 0.0);
        var d = 0.0;
        var vis = 0.0;
        if abs(anisotropy) > 0.0 {
            d = d_ggx_anisotropic(at, ab, dot(tangent, h), dot(bitangent, h), n_dot_h);
            vis = v_smith_anisotropic(
                at, ab,
                dot(tangent, v), dot(bitangent, v), dot(tangent, l), dot(bitangent, l),
                n_dot_v, n_dot_l,
            );
        } else {
            d = d_ggx(n_dot_h, alpha);
            vis = v_smith_correlated(n_dot_v, n_dot_l, alpha);
        }
        let f = f_schlick(f0, l_dot_h);
        let specular = d * vis * f * energy_compensation;
        let diffuse = diffuse_color / PI;
        var lobe = (diffuse + specular) * n_dot_l;

        if clearcoat > 0.0 {
            // Filament's `clearCoatLobe`: the lacquer's own highlight on the geometric normal,
            // and its Fresnel taking that much away from what is under it.
            let coat_alpha = clearcoat_roughness * clearcoat_roughness;
            let coat_n_dot_h = max(dot(n_coat, h), 0.0);
            let coat_n_dot_l = max(dot(n_coat, l), 0.0);
            let fc = f_schlick_scalar(0.04, l_dot_h) * clearcoat;
            let coat = d_ggx(coat_n_dot_h, coat_alpha) * v_kelemen(l_dot_h) * fc;
            lobe = lobe * (1.0 - fc) + vec3<f32>(coat * coat_n_dot_l);
        }
        var reaching = 1.0;
        if i == 0u {
            reaching = s.lit;
        }
        color = color + lobe * radiance * reaching;
    }

    if clearcoat > 0.0 {
        // Filament's `evaluateClearCoatIBL`: the coat's Fresnel attenuates everything beneath
        // it, and the coat's own reflection is the prefiltered environment weighted by that
        // Fresnel alone. Not by the BRDF table as well — that counted the Fresnel twice, and
        // left the coat forty times too dim head-on, so what showed as its reflection was the
        // flaked base underneath.
        let coat_n_dot_v = max(dot(n_coat, v), 1e-4);
        let fc = f_schlick_scalar(0.04, coat_n_dot_v) * clearcoat;
        let coat_reflected = reflect(-v, n_coat);
        let coat_ibl = sample_specular(coat_reflected, ibl_clearcoat_roughness);
        color = color * (1.0 - fc) + coat_ibl * (fc * env_intensity * s.open);
    }

    return color;
}

// AgX, in the minimal form: an inset toward the primaries, a log encoding over the
// curve's 16.5 stops, a sixth-order fit of its sigmoid, and the inverse inset. Chosen over ACES
// because it desaturates highlights less objectionably on saturated paint, which is the case
// this viewport is judged on (shading-spec.md §6.4). Returns linear light for the display's
// own transfer function to encode.
fn agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = inset * max(color, vec3<f32>(1e-10));
    x = clamp(log2(x), vec3<f32>(min_ev), vec3<f32>(max_ev));
    x = (x - min_ev) / (max_ev - min_ev);
    let x2 = x * x;
    let x4 = x2 * x2;
    x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2
        + 0.1191 * x - 0.00232;
    x = outset * x;
    // The curve's output is encoded with a 2.2 power; take it back to linear so the exact sRGB
    // transfer can be applied on store.
    return pow(max(x, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {
    let c = clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

// Whether this pixel is on the selection's outline: not of the selected body itself, but
// within the outline's radius of a pixel that is. Drawn outside the body rather than as a tint
// on it, so every pixel of a selected body keeps its own colour and a material can be judged
// while it is selected — which a yellow tint made impossible. Its silhouette against the
// background and its edges against other bodies both count, because both are where it ends.
//
// The cost is a few reads of the owner buffer per pixel and nothing of the field: for a radius
// of two, twenty-one 4-byte reads, against the hundreds of field evaluations the march spent.
fn selection_edge(pixel: vec2<u32>, owner: u32) -> f32 {
    let selected = shading.frame.x;
    if selected == 0xffffffffu || owner == selected {
        return 0.0;
    }
    let radius = i32(shading.frame.z);
    let width = i32(camera.frame.x);
    let height = i32(camera.frame.y);
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let x = i32(pixel.x) + dx;
            let y = i32(pixel.y) + dy;
            if x < 0 || y < 0 || x >= width || y >= height {
                continue;
            }
            if owners[u32(y * width + x)] == selected {
                return 1.0;
            }
        }
    }
    return 0.0;
}

// The draft read, as surface-analysis tools evaluate it: a face's angle from vertical along the pull, against
// the angle it needs to leave the mould. A face with enough draft is one colour; one without —
// vertical, or undercut — is the other; and the line between is blurred over a tolerance either
// side of the angle, so on a curved face it is a soft band rather than a contour that would
// read as a crease in the surface. The other mould half is the same read with the pull reversed.
fn draft_tint(n: vec3<f32>) -> vec3<f32> {
    let draft = degrees(asin(clamp(dot(n, shading.draft_dir.xyz), -1.0, 1.0)));
    let angle = shading.draft_params.x;
    let tolerance = max(shading.draft_params.y, 0.01);
    let enough = smoothstep(angle - tolerance, angle + tolerance, draft);
    return mix(shading.draft_fail.rgb, shading.draft_pass.rgb, enough);
}

// How open the surface at this pixel is and how much of the key light reaches it: read where the
// march measured them, and otherwise filled in from the measured pixels about it (loop spec P5).
//
// **A bilateral fill, by the surface rather than by the depth.** A still marches the occlusion and
// the shadow for one pixel in each two-by-two block, so every other pixel lies between two or four
// that were measured, a pixel apart. Those count where they are the same surface as this one
// (`one_surface`), and they count equally, since they stand at equal distances. Only a pixel
// `occlude` marked `NO_READING` is filled, which it does where two or more of those agree; any
// other was marched as well, and holds its own reading as every measured pixel does.
//
// So the nearest below is only for a pixel `occlude` judged fillable and this pass, from the same
// numbers in another pipeline, does not -- a neighbour on the boundary of `one_surface` by a
// rounding, which is the nearest one to the plane anyway. Where nothing measured hit at all, the
// surface reads as open and lit, which is also what a still reads with no `occlude` to measure it.
// The measured pixel moves round its block from frame to frame (`MEASURED_CORNERS` in render.rs),
// so every pixel is measured itself in one frame of each four and the still's average holds that.
fn measured_light(pixel: vec2<u32>, index: u32, p: vec3<f32>, n: vec3<f32>, t: f32) -> vec2<f32> {
    let own = unpack2x16float(extras[index]);
    if own.x >= 0.0 {
        return own;
    }
    let measured = occlusion_pixels(camera);
    let width = camera.frame.x;
    let across = bracket(pixel.x, measured.offset.x, width);
    let down = bracket(pixel.y, measured.offset.y, camera.frame.y);
    let reach = pixel_footprint(camera, t);
    var sum = vec2<f32>(0.0);
    var count = 0.0;
    var nearest = vec2<f32>(1.0);
    var nearest_gap = 3.4e38;
    for (var j = 0u; j < down.z; j = j + 1u) {
        let y = select(down.x, down.y, j == 1u);
        for (var i = 0u; i < across.z; i = i + 1u) {
            let x = select(across.x, across.y, i == 1u);
            let at = y * width + x;
            let found = gbuffer[at];
            if found.w < 0.0 {
                continue;
            }
            let gap = plane_gap(camera, p, n, vec2<u32>(x, y), found);
            let value = unpack2x16float(extras[at]);
            if gap < nearest_gap {
                nearest_gap = gap;
                nearest = value;
            }
            if one_surface(gap, reach, n, found) {
                sum = sum + value;
                count = count + 1.0;
            }
        }
    }
    if count > 0.0 {
        return sum / count;
    }
    return nearest;
}

@compute @workgroup_size(8, 8, 1)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = camera.frame.x;
    let height = camera.frame.y;
    // The band to light, which trails the one the march just wrote by the outline's reach --
    // see `present_rows`. Packed into one word: the first row, then how many.
    let lit_first = shading.frame.w & 0xffffu;
    let lit_count = shading.frame.w >> 16u;
    let row = gid.y + lit_first;
    if gid.x >= width || gid.y >= lit_count || row >= height {
        return;
    }
    let pixel = vec2<u32>(gid.x, row);
    let index = row * width + gid.x;

    let g = gbuffer[index];
    let hit = g.w >= 0.0;
    // Backlog item 116: the normal painted instead of lit. Taken before the shade rather than
    // after it, so the lights, the dome, the clearcoat and the occlusion read are skipped and not
    // merely zeroed -- which is where this view being cheaper than a lit one comes from.
    let normals = shading.look.w > 0.5;
    var color = shading.background.rgb;
    if hit {
        if normals {
            // The ordinary mapping, so +Z reads mid-blue and a face turning away darkens: the
            // same picture Fidget's viewer draws and this viewport drew before the lighting pass.
            color = 0.5 + 0.5 * normalize(g.xyz);
        } else {
            let ray = primary_ray(camera, pixel);
            let p = ray.origin + ray.dir * g.w;
            let n = normalize(g.xyz);
            let packed = measured_light(pixel, index, p, n, g.w);
            color = shade_surface(Surface(
                p,
                n,
                -ray.dir,
                packed.x,
                packed.y,
                pixel_footprint(camera, g.w),
            ));
        }
    }
    // The outline reads the owners around this pixel. Every row it can reach has been marched
    // this frame: the band being lit stops short of the march's by the outline's radius.
    let edge = selection_edge(pixel, owners[index]);

    // A still accumulates: each frame is one more jittered sample of the same view, and the
    // running mean is the anti-aliased image. The outline's coverage rides along in the alpha,
    // so it is anti-aliased the same way instead of shimmering as the samples move. The first
    // frame simply writes.
    let accumulated = shading.frame.y;
    var mean = vec4<f32>(color, edge);
    if accumulated > 0u {
        mean = mix(hdr[index], mean, 1.0 / (f32(accumulated) + 1.0));
    }
    hdr[index] = mean;

    var display = mean.rgb;
    if hit && !normals {
        display = agx(mean.rgb * shading.look.y);
    }
    // Analysis and interface over the finished image, in display space where exposure cannot
    // reach them: the draft read first, the selection outline over everything.
    // **A normal is data, not light, so it is written as it is.** Exposure and AgX are skipped
    // above for the same reason, and the sRGB encode has to go with them: it would put mid-grey
    // at 0.74 and bend the ramp, so two faces 10 degrees apart would read further apart near
    // grazing than face on -- which is exactly the judgement this view exists to support. A miss
    // keeps the encode, because the background behind the silhouette is still a colour.
    var encoded = select(encode_srgb(display), display, hit && normals);
    if hit && shading.draft_dir.w > 0.5 {
        encoded = mix(encoded, draft_tint(normalize(g.xyz)), shading.draft_params.z);
    }
    encoded = mix(encoded, shading.outline.rgb, mean.a);
    textureStore(present, vec2<i32>(pixel), vec4<f32>(encoded, 1.0));
}
