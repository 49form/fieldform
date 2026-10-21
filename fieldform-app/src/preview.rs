// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Drawing a tessellation in the viewport — backlog item 53's preview.
//!
//! **Its own pass with its own depth buffer, because the existing raster path has none.**
//! `render.rs` draws the control cage with back-face culling and nothing else, which is right for
//! a convex cage and wrong for a lattice: a mesh that hides most of itself behind itself needs a
//! depth test, and egui's pass has no depth attachment to test against. So the mesh is drawn into
//! an image of its own the size of the viewport, with a depth buffer, and that image is laid over
//! the viewport — the same compositing the overlay mode already does with the cage.
//!
//! The owner's rule was "whatever is faster for the viewport and the app": the mesh is drawn
//! *instead of* its body, which is taken out of the ray-trace while its mesh is on show. So a
//! lattice's preview costs a raster of triangles — the cheapest thing a GPU does — rather than the
//! march it replaces.

use std::collections::HashMap;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use nalgebra::{Matrix4, Vector3};
use wgpu::util::DeviceExt;

/// The most triangles a preview will upload.
///
/// **The GPU's own limit, not a taste.** Positions and indices are read as storage buffers, and
/// a storage binding is at most 128 MiB on the device's default limits — twelve bytes a triangle
/// of indices puts the ceiling near eleven million. Past that the block says the mesh is too big
/// to show rather than drawing a corrupt one; the file it writes is unaffected.
pub const MAX_PREVIEW_TRIANGLES: usize = 10_000_000;

/// One uploaded mesh.
struct Uploaded {
    /// Which of the tessellation's meshes this is, so a newer one replaces it.
    generation: u64,
    triangles: u32,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
}

/// The image the meshes are drawn into, sized to the viewport.
struct Target {
    size: [u32; 2],
    colour: wgpu::TextureView,
    depth: wgpu::TextureView,
    blit: wgpu::BindGroup,
}

/// Everything the preview needs on the GPU, kept in egui's callback resources.
pub struct PreviewResources {
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_layout: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    meshes: HashMap<u64, Uploaded>,
    target: Option<Target>,
}

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const UNIFORM_FLOATS: usize = 16 + 4 + 4;

impl PreviewResources {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/preview.wgsl").into()),
        });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mesh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-mesh"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
            ],
        });
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview-mesh"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("preview-mesh"),
                bind_group_layouts: &[Some(&mesh_layout)],
                immediate_size: 0,
            })),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Both sides: an open mesh, or a lattice seen from inside a cell, still has to
                // be readable, and the depth test is what does the hiding now.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview-blit"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("preview-blit"),
                bind_group_layouts: &[Some(&blit_layout)],
                immediate_size: 0,
            })),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_blit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preview"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            mesh_pipeline,
            mesh_layout,
            blit_pipeline,
            blit_layout,
            sampler,
            format,
            meshes: HashMap::new(),
            target: None,
        }
    }

    /// Whether `key`'s mesh at `generation` is already on the GPU.
    pub fn has(&self, key: u64, generation: u64) -> bool {
        self.meshes.get(&key).is_some_and(|m| m.generation == generation)
    }

    /// Puts a mesh on the GPU under `key`, replacing an older one.
    ///
    /// **Once per finished tessellation, never per frame** — a Full mesh is some thirty
    /// megabytes, and sending it again every frame would cost more than drawing it.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        key: u64,
        generation: u64,
        positions: &[nalgebra::Point3<f64>],
        triangles: &[[u32; 3]],
    ) {
        // The caller skips empty meshes; this is the belt to those braces, because the failure
        // it prevents is a zero-sized binding, which wgpu treats as fatal to the process.
        if triangles.is_empty() || positions.is_empty() {
            return;
        }
        let flat: Vec<f32> = positions
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let indices: Vec<u32> = triangles.iter().flatten().copied().collect();
        let storage = wgpu::BufferUsages::STORAGE;
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview-positions"),
            contents: bytemuck::cast_slice(&flat),
            usage: storage,
        });
        let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: storage,
        });
        // Its own uniforms, for the reason `render.rs` gives its meshes their own: every
        // callback's prepare runs before any paint, so two meshes sharing one buffer would both
        // draw with whichever wrote last.
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview-uniforms"),
            contents: bytemuck::cast_slice(&[0f32; UNIFORM_FLOATS]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-mesh"),
            layout: &self.mesh_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: indices_buffer.as_entire_binding() },
            ],
        });
        self.meshes.insert(
            key,
            Uploaded { generation, triangles: triangles.len() as u32, bind_group, uniforms },
        );
    }

    /// Drops every mesh whose key is not in `keep`.
    pub fn retain(&mut self, keep: &[u64]) {
        self.meshes.retain(|k, _| keep.contains(k));
    }

    fn ensure_target(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.target.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview-colour"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview-depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let colour = colour.create_view(&wgpu::TextureViewDescriptor::default());
        let blit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&colour),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.target = Some(Target {
            size,
            colour,
            depth: depth.create_view(&wgpu::TextureViewDescriptor::default()),
            blit,
        });
    }
}

/// One frame of the preview: which meshes to draw, from where.
pub struct PreviewCallback {
    pub mvp: Matrix4<f32>,
    /// Towards the eye, in world space — the light's direction.
    pub towards_eye: Vector3<f32>,
    /// The viewport's size in physical pixels, which the offscreen image has to match.
    pub size: [u32; 2],
    /// `(key, colour)` for each mesh on show.
    pub meshes: Vec<(u64, [f32; 3])>,
}

impl egui_wgpu::CallbackTrait for PreviewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = resources.get_mut::<PreviewResources>() else { return Vec::new() };
        if self.size[0] == 0 || self.size[1] == 0 {
            return Vec::new();
        }
        resources.ensure_target(device, self.size);
        for (key, colour) in &self.meshes {
            if let Some(mesh) = resources.meshes.get(key) {
                let mut data = [0f32; UNIFORM_FLOATS];
                data[..16].copy_from_slice(self.mvp.as_slice());
                data[16..19].copy_from_slice(self.towards_eye.as_slice());
                data[20..23].copy_from_slice(colour);
                data[23] = 1.0;
                queue.write_buffer(&mesh.uniforms, 0, bytemuck::cast_slice(&data));
            }
        }
        let target = resources.target.as_ref().expect("made just above");
        // Recorded into egui's own encoder, ahead of egui's pass, so the image is finished
        // before anything samples it.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("preview"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.colour,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&resources.mesh_pipeline);
        for (key, _) in &self.meshes {
            if let Some(mesh) = resources.meshes.get(key) {
                pass.set_bind_group(0, &mesh.bind_group, &[]);
                pass.draw(0..mesh.triangles * 3, 0..1);
            }
        }
        drop(pass);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(resources) = resources.get::<PreviewResources>() else { return };
        let Some(target) = &resources.target else { return };
        render_pass.set_pipeline(&resources.blit_pipeline);
        render_pass.set_bind_group(0, &target.blit, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    /// **The preview's shader is valid WGSL**, checked here because the application finds out at
    /// the moment it first draws a mesh — and wgpu's answer to a shader it cannot compile is to
    /// panic the process, taking the document with it.
    #[test]
    fn the_preview_shader_is_valid_wgsl() {
        let source = include_str!("shaders/preview.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }
}
