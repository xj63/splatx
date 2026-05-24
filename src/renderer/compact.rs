use wgpu::include_wgsl;

use super::{
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer, layout_entry},
};

const WORKGROUP_SIZE: u32 = 64;

pub struct CompactStage {
    uniform: wgpu::Buffer,
    alive_indices: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    len: u32,
}

impl CompactStage {
    pub fn new(
        device: &wgpu::Device,
        len: u32,
        mask: &wgpu::Buffer,
        prefix: &wgpu::Buffer,
    ) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx compact uniform"),
            size: std::mem::size_of::<CompactUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let alive_indices =
            create_storage_buffer(device, "splatx compact alive indices", len as usize * 4);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx compact bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/compact.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx compact pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx compact pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx compact bind group"),
            layout: &bind_group_layout,
            entries: &[
                bind_entry(0, &uniform),
                bind_entry(1, mask),
                bind_entry(2, prefix),
                bind_entry(3, &alive_indices),
            ],
        });

        Self {
            uniform,
            alive_indices,
            bind_group,
            pipeline,
            len,
        }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        profiler: &mut GpuProfilerFrame<'_>,
    ) {
        let uniform = CompactUniform {
            values: [self.len, 0, 0, 0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compact"),
            timestamp_writes: None,
        });
        profiler.profile_compute(&mut pass, "Compact", |pass| {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.len.div_ceil(WORKGROUP_SIZE), 1, 1);
        });
    }

    pub fn alive_indices(&self) -> &wgpu::Buffer {
        &self.alive_indices
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompactUniform {
    values: [u32; 4],
}
