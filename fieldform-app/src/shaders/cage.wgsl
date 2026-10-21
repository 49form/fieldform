// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
struct Uniforms {
    mvp: mat4x4<f32>,
    // rgb = base colour, a = opacity (the cage overlay draws semi-transparent over the field)
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.5));
    // The underside of an open surface is lit as the side it is. A cage never shows a back face
    // -- its pipeline culls them -- so this only ever turns a surface seen from below.
    let n = select(-normalize(in.normal), normalize(in.normal), front);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.25;
    let color = uniforms.color.rgb * (ambient + diffuse * 0.75);
    // Premultiplied alpha, matching the pipeline's blend state.
    return vec4<f32>(color * uniforms.color.a, uniforms.color.a);
}
