use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use super::sort::SortStage;

pub struct RenderStage {
    uniform: wgpu::Buffer,
    bind_group_layout0: wgpu::BindGroupLayout,
    bind_group_layout1: wgpu::BindGroupLayout,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
    shader: wgpu::ShaderModule,
    pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    draw_args: wgpu::Buffer,
}

impl RenderStage {
    pub fn new(
        device: &wgpu::Device,
        projected_splats: &wgpu::Buffer,
        sorted_indices: &wgpu::Buffer,
        draw_args: &wgpu::Buffer,
    ) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx render uniform"),
            size: std::mem::size_of::<RenderUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("splatx render bind group layout 0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bind_group_layout1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("splatx render bind group layout 1"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx render bind group 0"),
            layout: &bind_group_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: projected_splats.as_entire_binding(),
                },
            ],
        });
        let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx render bind group 1"),
            layout: &bind_group_layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sorted_indices.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/gaussian.wgsl"));

        Self {
            uniform,
            bind_group_layout0,
            bind_group_layout1,
            bind_group0,
            bind_group1,
            shader,
            pipeline: None,
            draw_args: draw_args.clone(),
        }
    }

    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        let uniform = RenderUniform {
            viewport: [width.max(1) as f32, height.max(1) as f32, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        if self
            .pipeline
            .as_ref()
            .map(|(current_format, _)| *current_format != format)
            .unwrap_or(true)
        {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("splatx render pipeline layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout0), Some(&self.bind_group_layout1)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("splatx gaussian render pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            self.pipeline = Some((format, pipeline));
        }

        let pipeline = &self.pipeline.as_ref().expect("render pipeline").1;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splatx gaussian render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group0, &[]);
        pass.set_bind_group(1, &self.bind_group1, &[]);
        pass.draw_indirect(&self.draw_args, 0);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderUniform {
    viewport: [f32; 4],
}
