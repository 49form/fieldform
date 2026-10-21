// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What rebuilding the cage's drawing resources costs, which the application does after every
//! edit that changes the document.
//!
//! **Written from a report of a half-second hang** after moving a cage face in Polygon view
//! (owner, 2026-09-19). The CPU side of a finished edit measures 2 ms
//! (`fieldform-field/examples/cage_edit_cost.rs`), so the question is what the GPU side costs:
//! `CageRenderResources::new` compiles `cage.wgsl` and creates a render pipeline from it — two
//! of them since an open surface needs one that culls nothing — and it is called afresh on every
//! rebuild rather than kept.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example cage_pipeline_cost`

fn main() {
    let (device, _queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("an adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("a device")
    });
    println!("{}\n", device.adapter_info().name);

    let source = include_str!("../../fieldform-app/src/shaders/cage.wgsl");
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;

    // Five rounds, because the first of anything on a driver pays for warming it and the
    // application's second edit does not.
    for round in 1..=5 {
        let t = std::time::Instant::now();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cage"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let module_ms = ms(t);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cage-uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cage"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let mut each = Vec::new();
        for cull in [Some(wgpu::Face::Back), None] {
            let t = std::time::Instant::now();
            let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cage"),
                layout: Some(&pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
            });
            each.push(ms(t));
        }
        // What the driver has actually finished, rather than what it has queued: pipeline
        // creation may report back long before the work is done.
        let t = std::time::Instant::now();
        device.poll(wgpu::PollType::wait_indefinitely()).expect("polls");
        let settled = ms(t);
        println!(
            "round {round}: shader module {module_ms:>7.1} ms, culled pipeline {:>7.1} ms, \
             two-sided {:>7.1} ms, waiting for the driver {settled:>7.1} ms  \u{2014} total {:>7.1} ms",
            each[0],
            each[1],
            module_ms + each[0] + each[1] + settled
        );
    }
}
