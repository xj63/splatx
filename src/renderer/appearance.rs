use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use crate::camera::Camera;

use super::{
    data::GpuModelData,
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer, layout_entry},
};

pub const APPEARANCE_WORKGROUP_SIZE: u32 = 64;

pub struct AppearanceStage {
    uniform: wgpu::Buffer,
    rgba: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    dispatch_args: wgpu::Buffer,
}

impl AppearanceStage {
    pub fn new(
        device: &wgpu::Device,
        len: u32,
        data: &GpuModelData,
        alive_indices: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
    ) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx appearance uniform"),
            size: std::mem::size_of::<AppearanceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rgba = create_storage_buffer(
            device,
            "splatx appearance rgba",
            len as usize * std::mem::size_of::<[f32; 4]>(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx appearance bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(6, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(7, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(8, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/appearance.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx appearance pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx appearance pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx appearance bind group"),
            layout: &bind_group_layout,
            entries: &[
                bind_entry(0, &uniform),
                bind_entry(1, &data.gaussians),
                bind_entry(2, &data.features_static),
                bind_entry(3, &data.features_view),
                bind_entry(4, &data.weights_cont),
                bind_entry(5, &data.weights_attr),
                bind_entry(6, alive_indices),
                bind_entry(7, dispatch_args),
                bind_entry(8, &rgba),
            ],
        });

        Self {
            uniform,
            rgba,
            bind_group,
            pipeline,
            dispatch_args: dispatch_args.clone(),
        }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        profiler: &mut GpuProfilerFrame<'_>,
        camera: &Camera,
        time: f32,
    ) {
        let uniform = AppearanceUniform {
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            params: [time, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Appearance"),
            timestamp_writes: None,
        });
        profiler.profile_compute(&mut pass, "Appearance", |pass| {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch_args, 0);
        });
    }

    pub fn rgba(&self) -> &wgpu::Buffer {
        &self.rgba
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AppearanceUniform {
    camera_position: [f32; 4],
    params: [f32; 4],
}
