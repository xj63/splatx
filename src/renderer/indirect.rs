use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use super::{
    profiler::GpuProfilerFrame,
    util::{bind_entry, layout_entry},
};

pub struct IndirectStage {
    dispatch_args: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl IndirectStage {
    pub fn new(
        device: &wgpu::Device,
        len: u32,
        workgroup_size: u32,
        mask: &wgpu::Buffer,
        prefix: &wgpu::Buffer,
    ) -> Self {
        let uniform_data = IndirectUniform {
            values: [len, workgroup_size, 0, 0],
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx indirect uniform"),
            size: std::mem::size_of::<IndirectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&uniform_data));
        uniform.unmap();
        let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx indirect dispatch args"),
            size: std::mem::size_of::<[u32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx indirect bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/indirect.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx indirect pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx indirect pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx indirect bind group"),
            layout: &bind_group_layout,
            entries: &[
                bind_entry(0, &uniform),
                bind_entry(1, mask),
                bind_entry(2, prefix),
                bind_entry(3, &dispatch_args),
            ],
        });

        Self {
            dispatch_args,
            bind_group,
            pipeline,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, profiler: &mut GpuProfilerFrame<'_>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Indirect"),
            timestamp_writes: None,
        });
        profiler.profile_compute(&mut pass, "Indirect", |pass| {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });
    }

    pub fn dispatch_args(&self) -> &wgpu::Buffer {
        &self.dispatch_args
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndirectUniform {
    values: [u32; 4],
}
