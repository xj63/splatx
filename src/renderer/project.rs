use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use crate::camera::Camera;

use super::{
    data::GpuModelData,
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer, layout_entry},
};

pub const PROJECT_WORKGROUP_SIZE: u32 = 64;
const KERNEL_SIZE: f32 = 0.3;

pub struct ProjectStage {
    uniform: wgpu::Buffer,
    projected: wgpu::Buffer,
    depths: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    dispatch_args: wgpu::Buffer,
}

impl ProjectStage {
    pub fn new(
        device: &wgpu::Device,
        len: u32,
        data: &GpuModelData,
        alive_indices: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
    ) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx project uniform"),
            size: std::mem::size_of::<ProjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let projected = create_storage_buffer(
            device,
            "splatx projected splats",
            len as usize * std::mem::size_of::<[[f32; 4]; 2]>(),
        );
        let depths = create_storage_buffer(
            device,
            "splatx projected depths",
            len as usize * std::mem::size_of::<f32>(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx project bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shader = device.create_shader_module(include_wgsl!("shader/project.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx project pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx project pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx project bind group"),
            layout: &bind_group_layout,
            entries: &[
                bind_entry(0, &uniform),
                bind_entry(1, &data.gaussians),
                bind_entry(2, &data.covariances),
                bind_entry(3, alive_indices),
                bind_entry(4, dispatch_args),
                bind_entry(5, &projected),
                bind_entry(6, &depths),
            ],
        });

        Self {
            uniform,
            projected,
            depths,
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
        width: u32,
        height: u32,
    ) {
        let aspect = width as f32 / height.max(1) as f32;
        let fy = height as f32 * 0.5 / (camera.fovy_radians * 0.5).tan();
        let fx = fy * aspect;
        let uniform = ProjectUniform {
            view: camera.view_matrix().to_cols_array_2d(),
            view_projection: camera.view_projection_matrix(aspect).to_cols_array_2d(),
            focal_time_kernel: [fx, fy, time, KERNEL_SIZE],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Project"),
            timestamp_writes: None,
        });
        profiler.profile_compute(&mut pass, "Project", |pass| {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch_args, 0);
        });
    }

    pub fn projected(&self) -> &wgpu::Buffer {
        &self.projected
    }

    pub fn depths(&self) -> &wgpu::Buffer {
        &self.depths
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProjectUniform {
    view: [[f32; 4]; 4],
    view_projection: [[f32; 4]; 4],
    focal_time_kernel: [f32; 4],
}
