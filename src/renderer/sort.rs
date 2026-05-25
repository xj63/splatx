use bytemuck::{Pod, Zeroable};
use wgpu::include_wgsl;

use super::{
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer, layout_entry},
};

const RADIX_SIZE: u32 = 256;
const SORT_BLOCK_SIZE: u32 = 256;
const PREPROCESS_WORKGROUP_SIZE: u32 = 64;

pub struct SortStage {
    histograms: wgpu::Buffer,
    keys_a: wgpu::Buffer,
    keys_b: wgpu::Buffer,
    payload_a: wgpu::Buffer,
    payload_b: wgpu::Buffer,
    bind_groups: Vec<wgpu::BindGroup>,
    preprocess_pipeline: wgpu::ComputePipeline,
    histogram_pipeline: wgpu::ComputePipeline,
    prefix_pipeline: wgpu::ComputePipeline,
    scatter_even_pipeline: wgpu::ComputePipeline,
    scatter_odd_pipeline: wgpu::ComputePipeline,
    len: u32,
    num_blocks: u32,
}

impl SortStage {
    pub fn new(
        device: &wgpu::Device,
        len: u32,
        depths: &wgpu::Buffer,
        alive_indices: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
    ) -> Self {
        let num_blocks = len.max(1).div_ceil(SORT_BLOCK_SIZE);
        let histograms = create_storage_buffer(
            device,
            "splatx sort histograms",
            num_blocks as usize * RADIX_SIZE as usize * std::mem::size_of::<u32>(),
        );
        let keys_a = create_storage_buffer(
            device,
            "splatx sort keys a",
            len as usize * std::mem::size_of::<u32>(),
        );
        let keys_b = create_storage_buffer(
            device,
            "splatx sort keys b",
            len as usize * std::mem::size_of::<u32>(),
        );
        let payload_a = create_storage_buffer(
            device,
            "splatx sort payload a",
            len as usize * std::mem::size_of::<u32>(),
        );
        let payload_b = create_storage_buffer(
            device,
            "splatx sort payload b",
            len as usize * std::mem::size_of::<u32>(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splatx sort bind group layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(8, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let shader = device.create_shader_module(include_wgsl!("shader/sort.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splatx sort pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let preprocess_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("splatx sort preprocess"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("preprocess"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx sort histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("histogram"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let prefix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx sort prefix"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("prefix_histograms"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scatter_even_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("splatx sort scatter even"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_even"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let scatter_odd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("splatx sort scatter odd"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("scatter_odd"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut bind_groups = Vec::with_capacity(4);
        for radix_pass in 0..4_u32 {
            let uniform_data = SortUniform {
                values: [len, num_blocks, SORT_BLOCK_SIZE, radix_pass],
            };
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("splatx sort uniform"),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            uniform
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&uniform_data));
            uniform.unmap();

            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splatx sort bind group"),
                layout: &bind_group_layout,
                entries: &[
                    bind_entry(0, &uniform),
                    bind_entry(1, dispatch_args),
                    bind_entry(2, depths),
                    bind_entry(3, alive_indices),
                    bind_entry(4, &histograms),
                    bind_entry(5, &keys_a),
                    bind_entry(6, &keys_b),
                    bind_entry(7, &payload_a),
                    bind_entry(8, &payload_b),
                ],
            }));
        }

        Self {
            histograms,
            keys_a,
            keys_b,
            payload_a,
            payload_b,
            bind_groups,
            preprocess_pipeline,
            histogram_pipeline,
            prefix_pipeline,
            scatter_even_pipeline,
            scatter_odd_pipeline,
            len,
            num_blocks,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, profiler: &mut GpuProfilerFrame<'_>) {
        let _ = (
            &self.histograms,
            &self.keys_a,
            &self.keys_b,
            &self.payload_b,
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sort"),
            timestamp_writes: None,
        });
        profiler.profile_compute(&mut pass, "Sort", |pass| {
            pass.set_bind_group(0, &self.bind_groups[0], &[]);

            pass.set_pipeline(&self.preprocess_pipeline);
            pass.dispatch_workgroups(self.len.div_ceil(PREPROCESS_WORKGROUP_SIZE).max(1), 1, 1);

            for radix_pass in 0..4_u32 {
                pass.set_bind_group(0, &self.bind_groups[radix_pass as usize], &[]);
                pass.set_pipeline(&self.histogram_pipeline);
                pass.dispatch_workgroups(self.num_blocks.max(1), 1, 1);

                pass.set_pipeline(&self.prefix_pipeline);
                pass.dispatch_workgroups(1, 1, 1);

                if radix_pass % 2 == 0 {
                    pass.set_pipeline(&self.scatter_even_pipeline);
                } else {
                    pass.set_pipeline(&self.scatter_odd_pipeline);
                }
                pass.dispatch_workgroups(self.num_blocks.max(1), 1, 1);
            }
        });
    }

    pub fn sorted_indices(&self) -> &wgpu::Buffer {
        &self.payload_a
    }

    pub fn sorted_keys(&self) -> &wgpu::Buffer {
        &self.keys_a
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortUniform {
    values: [u32; 4],
}
