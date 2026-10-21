// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! wgpu rendering for the viewport, wired into egui via a [`egui_wgpu::CallbackTrait`] paint
//! callback (the pattern eframe/egui's own `custom3d_wgpu` example uses for embedding custom
//! wgpu draws inside an egui panel).
//!
//! The control cage is the only mesh here. Overlay mode draws it translucent over the traced
//! field, which is not a mesh at all — it is evaluated per pixel by `traced`.
//!
//! Not yet depth-tested — correctness currently relies on back-face culling plus a convex
//! shape. This will need a real depth buffer once editing can produce concave cages (see
//! project plan, M1).

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

use crate::cage_mesh::{CageMesh, Vertex};

/// Standard OpenGL-clip-space (z in [-1, 1]) to wgpu clip-space (z in [0, 1]) correction,
/// applied on top of a conventional nalgebra perspective matrix.
#[rustfmt::skip]
const OPENGL_TO_WGPU: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub fn opengl_to_wgpu_projection() -> Matrix4<f32> {
    OPENGL_TO_WGPU
}

/// Vertex/index buffers for the mesh, plus its uniform buffer and bind group.
///
/// **A second mesh drawn in the same frame needs its own uniform buffer**, not another draw
/// against this one. egui_wgpu runs *every* callback's `prepare` before *any* callback's
/// `paint`, so two callbacks sharing one uniform buffer both paint with whichever wrote last
/// — which is how overlay mode once drew the cage and the limit surface in identical colours.
/// There is one mesh here today only because the viewport draws exactly one: the control cage.
struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Drawn through the pipeline that culls nothing — see `CageMesh::two_sided`.
    two_sided: bool,
}

/// Uniform block layout, matching `cage.wgsl`: a 4x4 matrix followed by an RGBA colour.
const UNIFORM_FLOATS: usize = 16 + 4;

/// GPU resources for the viewport, stored in egui_wgpu's `CallbackResources` type map so they
/// live as long as the render pass.
pub struct CageRenderResources {
    pipeline: wgpu::RenderPipeline,
    /// The same pipeline culling nothing, for an open surface drawn as the control mesh.
    two_sided_pipeline: wgpu::RenderPipeline,
    mesh: MeshBuffers,
    /// Imported geometry, drawn through the same pipeline.
    ///
    /// A second buffer rather than a second resource because egui's `CallbackResources` is a
    /// type map — two values of one type cannot both live in it — and a second *type* would be
    /// a copy of this one differing only in its name. The pipeline is identical: triangles
    /// with a flat normal, which is all either of them is.
    imported: MeshBuffers,
}

/// Which mesh a draw is for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DrawTarget {
    /// The control cage, drawn through its own placement.
    Cage,
    /// Imported geometry, whose placements are already baked into its vertices because
    /// several imports can sit under different ones and a single MVP cannot say that.
    Imported,
}

impl CageRenderResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        cage_mesh: &CageMesh,
        imported_mesh: &CageMesh,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cage"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cage.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cage"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let pipeline_culling = |cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: target_format,
                    // Premultiplied-alpha blending, so a translucent cage can be drawn over
                    // the field. Opaque draws simply pass alpha = 1.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            })
        };
        let pipeline = pipeline_culling(Some(wgpu::Face::Back));
        let two_sided_pipeline = pipeline_culling(None);

        let make_mesh = |mesh: &CageMesh, label: &str| -> MeshBuffers {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-vertices")),
                contents: bytemuck::cast_slice(&mesh.vertices),
                // COPY_DST so an edit can overwrite the few vertices it changed rather than
                // reallocating the whole buffer — see `write_vertices`.
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-indices")),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-uniforms")),
                contents: bytemuck::cast_slice(&[0f32; UNIFORM_FLOATS]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{label}-uniforms")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            MeshBuffers {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                uniform_buffer,
                bind_group,
                two_sided: mesh.two_sided,
            }
        };

        Self {
            pipeline,
            two_sided_pipeline,
            mesh: make_mesh(cage_mesh, "cage"),
            imported: make_mesh(imported_mesh, "imported"),
        }
    }

    fn buffers(&self, target: DrawTarget) -> &MeshBuffers {
        match target {
            DrawTarget::Cage => &self.mesh,
            DrawTarget::Imported => &self.imported,
        }
    }

    /// Overwrites a contiguous run of vertices in one mesh's buffer.
    ///
    /// `first_vertex` is an index into the mesh's vertex array, not a byte offset.
    ///
    /// This exists because recreating the buffers on every drag frame made the app's cost
    /// scale with model size even after tessellation stopped doing so — a 100k-face cage
    /// meant re-uploading megabytes to change a handful of vertices. Writing only the changed
    /// range is what makes the locality work visible in the app rather than only in a
    /// benchmark.
    ///
    /// Returns whether the write was in range; a caller that has changed the vertex *count*
    /// must rebuild the resources instead, since the buffer is fixed-size.
    pub fn write_vertices(
        &self,
        queue: &wgpu::Queue,
        first_vertex: usize,
        vertices: &[Vertex],
    ) -> bool {
        let mesh = &self.mesh;
        let stride = std::mem::size_of::<Vertex>();
        let end = first_vertex + vertices.len();
        if (end * stride) as u64 > mesh.vertex_buffer.size() {
            return false;
        }
        queue.write_buffer(
            &mesh.vertex_buffer,
            (first_vertex * stride) as u64,
            bytemuck::cast_slice(vertices),
        );
        true
    }

    fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        target: DrawTarget,
        mvp: &Matrix4<f32>,
        color: [f32; 4],
    ) {
        let mut data = [0f32; UNIFORM_FLOATS];
        data[..16].copy_from_slice(mvp.as_slice());
        data[16..].copy_from_slice(&color);
        queue.write_buffer(&self.buffers(target).uniform_buffer, 0, bytemuck::cast_slice(&data));
    }

    fn paint(&self, target: DrawTarget, render_pass: &mut wgpu::RenderPass<'_>) {
        let mesh = self.buffers(target);
        if mesh.index_count == 0 {
            return;
        }
        render_pass.set_pipeline(if mesh.two_sided {
            &self.two_sided_pipeline
        } else {
            &self.pipeline
        });
        render_pass.set_bind_group(0, &mesh.bind_group, &[]);
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
    }
}

/// One draw of the control cage: opaque in cage mode, translucent over the traced field in
/// overlay mode.
pub struct CageDrawCallback {
    pub mvp: Matrix4<f32>,
    /// rgb + opacity.
    pub color: [f32; 4],
    pub target: DrawTarget,
}

impl egui_wgpu::CallbackTrait for CageDrawCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &CageRenderResources = resources.get().unwrap();
        resources.write_uniforms(queue, self.target, &self.mvp, self.color);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &CageRenderResources = resources.get().unwrap();
        resources.paint(self.target, render_pass);
    }
}
