// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
// A tessellation drawn so its triangles can be seen — backlog item 53's preview.
//
// Flat-shaded, with every triangle's edges drawn in the same pass: the whole point of looking is
// to see whether the detail chosen was fine enough, and that is a question about the triangles
// rather than about the surface they approximate.
//
// **No vertex buffer.** The vertex shader reads the mesh's own index and position arrays, one
// triangle corner per invocation, so each corner knows which corner it is and can hand the
// fragment shader a barycentric coordinate. That is what draws the edges without a second pass,
// a line primitive, or tripling the vertex data — at Full detail a mesh is two million triangles,
// and a wireframe drawn as separate lines would be three million more primitives to raster.

struct Uniforms {
    mvp: mat4x4<f32>,
    // The direction towards the eye, so the light comes from where the person is looking.
    light: vec4<f32>,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct Corner {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) bary: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> Corner {
    let corner = i % 3u;
    let v = indices[i];
    let p = vec3<f32>(positions[v * 3u], positions[v * 3u + 1u], positions[v * 3u + 2u]);
    var out: Corner;
    out.clip = u.mvp * vec4<f32>(p, 1.0);
    out.world = p;
    out.bary = vec3<f32>(
        select(0.0, 1.0, corner == 0u),
        select(0.0, 1.0, corner == 1u),
        select(0.0, 1.0, corner == 2u),
    );
    return out;
}

@fragment
fn fs_main(in: Corner) -> @location(0) vec4<f32> {
    // The triangle's own normal, from how its position changes across the screen — flat by
    // construction, which is what makes the facets visible as facets.
    let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    // Two-sided: a mesh seen from inside a lattice cell is still a mesh worth reading.
    let lit = abs(dot(n, normalize(u.light.xyz)));
    let shade = u.colour.rgb * (0.28 + 0.72 * lit);

    // One pixel of edge, measured in screen space so it stays one pixel at any zoom.
    let w = fwidth(in.bary);
    let near_edge = smoothstep(vec3<f32>(0.0), w * 1.5, in.bary);
    let edge = 1.0 - min(min(near_edge.x, near_edge.y), near_edge.z);
    // **Faded out where the triangles are smaller than the lines between them.** Past that the
    // edges are all there is to see and the part turns into a solid dark smear; fading them is
    // what keeps a zoomed-out Enchilada mesh readable and a zoomed-in one crisp.
    let crowded = max(max(w.x, w.y), w.z);
    let fade = 1.0 - smoothstep(0.12, 0.35, crowded);
    return vec4<f32>(mix(shade, shade * 0.3, edge * fade), 1.0);
}

// ---- laying the offscreen image over the viewport ----

// Bindings of their own rather than 0 and 1 again: one module declares both passes, and two
// resource variables on the same slot are refused whatever entry point reads them.
@group(0) @binding(3) var image: texture_2d<f32>;
@group(0) @binding(4) var image_sampler: sampler;

struct Screen {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the viewport, which egui has already set to the callback's rect.
@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> Screen {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Screen;
    out.clip = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

@fragment
fn fs_blit(in: Screen) -> @location(0) vec4<f32> {
    // Premultiplied already: the mesh pass clears to transparent black and writes opaque
    // colour, so where there is no mesh this is zero and the traced image shows through.
    return textureSample(image, image_sampler, in.uv);
}
