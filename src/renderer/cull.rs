use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use super::{
    data::GpuModelData,
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer, layout_entry},
};

pub struct CullStage {
    uniform: wgpu::Buffer,
    mask: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    len: u32,
}

pub struct CullParams {
    pub view_projection: [f32; 16],
    pub time: f32,
}

impl CullStage {
    pub fn new(device: &wgpu::Device, data: &GpuModelData, len: u32) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx cull uniform"),
            size: std::mem::size_of::<CullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask = create_storage_buffer(device, "splatx cull mask", len as usize * 4);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx cull bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx cull bind group"),
            layout: &layout,
            entries: &[
                bind_entry(0, &uniform),
                bind_entry(1, &data.gaussians),
                bind_entry(2, &mask),
            ],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/cull.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx cull pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx cull pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            uniform,
            mask,
            bind_group,
            pipeline,
            len,
        }
    }

    pub fn mask(&self) -> &wgpu::Buffer {
        &self.mask
    }
}

impl CullStage {
    pub fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        profiler: &mut GpuProfilerFrame<'_>,
        params: CullParams,
    ) {
        let uniform = CullUniform {
            view_projection: params.view_projection,
            time: params.time,
            gaussian_count: self.len,
            _padding: [0; 2],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cull"),
                timestamp_writes: None,
            });
            profiler.profile_compute(&mut pass, "Cull", |pass| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(self.len.div_ceil(64), 1, 1);
            });
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniform {
    view_projection: [f32; 16],
    time: f32,
    gaussian_count: u32,
    _padding: [u32; 2],
}
