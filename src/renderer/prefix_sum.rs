use super::{
    profiler::GpuProfilerFrame,
    util::{bind_entry, create_storage_buffer},
};
use wgpu::include_wgsl;

// Must match WG_SIZE in prefix_sum_block_scan.wgsl and prefix_sum_add_carry.wgsl.
const WG_SIZE: u32 = 128;

pub struct PrefixSumStage {
    prefix: wgpu::Buffer,
    block_sums: Vec<wgpu::Buffer>,
    pipeline_write_sum: wgpu::ComputePipeline,
    pipeline_no_sum: wgpu::ComputePipeline,
    pipeline_add_carry: wgpu::ComputePipeline,
    bind_groups_write_sum: Vec<wgpu::BindGroup>,
    bind_group_no_sum: wgpu::BindGroup,
    bind_groups_add_carry: Vec<wgpu::BindGroup>,
    level_lengths: Vec<u32>,
    max_dispatch_dim: u32,
    len: u32,
}

impl PrefixSumStage {
    pub fn new(device: &wgpu::Device, len: u32) -> Self {
        let prefix = create_storage_buffer(device, "splatx prefix scan data", len as usize * 4);
        let has_subgroup = device.features().contains(wgpu::Features::SUBGROUP);
        let block_scan_shader = if has_subgroup {
            device.create_shader_module(include_wgsl!("shader/prefix_sum_block_scan.wgsl"))
        } else {
            tracing::info!("subgroup feature unavailable; using fallback prefix-sum stage");
            device.create_shader_module(include_wgsl!("shader/prefix_sum_block_scan_fallback.wgsl"))
        };
        let add_carry_shader = if has_subgroup {
            device.create_shader_module(include_wgsl!("shader/prefix_sum_add_carry.wgsl"))
        } else {
            device.create_shader_module(include_wgsl!("shader/prefix_sum_add_carry_fallback.wgsl"))
        };

        let pipeline_write_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx prefix block_scan_write_sum"),
            layout: None,
            module: &block_scan_shader,
            entry_point: Some("block_scan_write_sum"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_no_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx prefix block_scan_no_sum"),
            layout: None,
            module: &block_scan_shader,
            entry_point: Some("block_scan_no_sum"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_add_carry = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splatx prefix add_carry"),
            layout: None,
            module: &add_carry_shader,
            entry_point: Some("add_carry"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut block_sums = Vec::new();
        let mut bind_groups_write_sum = Vec::new();
        let mut bind_groups_add_carry = Vec::new();
        let mut level_lengths = Vec::new();

        let mut current_len = len.max(1);
        let mut current_buffer = &prefix;
        while current_len > WG_SIZE {
            level_lengths.push(current_len);
            let num_blocks = current_len.div_ceil(WG_SIZE).max(1);
            let next_buffer = create_storage_buffer(
                device,
                "splatx prefix scan block sums",
                num_blocks as usize * 4,
            );

            bind_groups_write_sum.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splatx prefix write-sum bind group"),
                layout: &pipeline_write_sum.get_bind_group_layout(0),
                entries: &[bind_entry(0, current_buffer), bind_entry(1, &next_buffer)],
            }));

            block_sums.push(next_buffer);
            current_buffer = block_sums.last().expect("block sum buffer");
            current_len = num_blocks;
        }
        level_lengths.push(current_len);

        let terminal_buffer: &wgpu::Buffer = block_sums.last().unwrap_or(&prefix);
        let bind_group_no_sum = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splatx prefix no-sum bind group"),
            layout: &pipeline_no_sum.get_bind_group_layout(0),
            entries: &[bind_entry(0, terminal_buffer)],
        });

        let mut data_chain = Vec::with_capacity(block_sums.len() + 1);
        data_chain.push(&prefix);
        for buffer in &block_sums {
            data_chain.push(buffer);
        }
        for level in (1..data_chain.len()).rev() {
            bind_groups_add_carry.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splatx prefix add-carry bind group"),
                layout: &pipeline_add_carry.get_bind_group_layout(0),
                entries: &[
                    bind_entry(0, data_chain[level - 1]),
                    bind_entry(1, data_chain[level]),
                ],
            }));
        }

        Self {
            prefix,
            block_sums,
            pipeline_write_sum,
            pipeline_no_sum,
            pipeline_add_carry,
            bind_groups_write_sum,
            bind_group_no_sum,
            bind_groups_add_carry,
            level_lengths,
            max_dispatch_dim: device.limits().max_compute_workgroups_per_dimension,
            len,
        }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut GpuProfilerFrame<'_>,
        mask: &wgpu::Buffer,
    ) {
        encoder.copy_buffer_to_buffer(mask, 0, &self.prefix, 0, self.len as u64 * 4);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PrefixSum"),
                timestamp_writes: None,
            });

            profiler.profile_compute(&mut pass, "PrefixSum", |pass| {
                pass.set_pipeline(&self.pipeline_write_sum);
                for (level, bind_group) in self.bind_groups_write_sum.iter().enumerate() {
                    let workgroups_needed = self.level_lengths[level].div_ceil(WG_SIZE).max(1);
                    pass.set_bind_group(0, bind_group, &[]);
                    let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dispatch_dim);
                    pass.dispatch_workgroups(x, y, z);
                }

                let terminal_level = self.level_lengths.len() - 1;
                let workgroups_needed = self.level_lengths[terminal_level].div_ceil(WG_SIZE).max(1);
                pass.set_pipeline(&self.pipeline_no_sum);
                pass.set_bind_group(0, &self.bind_group_no_sum, &[]);
                let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dispatch_dim);
                pass.dispatch_workgroups(x, y, z);

                pass.set_pipeline(&self.pipeline_add_carry);
                for (index, bind_group) in self.bind_groups_add_carry.iter().enumerate() {
                    let level = self.block_sums.len() - index;
                    let workgroups_needed = self.level_lengths[level - 1].div_ceil(WG_SIZE).max(1);
                    pass.set_bind_group(0, bind_group, &[]);
                    let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dispatch_dim);
                    pass.dispatch_workgroups(x, y, z);
                }
            });
        }
    }

    pub fn prefix(&self) -> &wgpu::Buffer {
        &self.prefix
    }
}

fn split_dispatch_3d(workgroups_needed: u32, max_dim: u32) -> [u32; 3] {
    let x = workgroups_needed.min(max_dim);
    let remaining = workgroups_needed.div_ceil(x);
    let y = remaining.min(max_dim);
    let xy = x as u64 * y as u64;
    let z = (workgroups_needed as u64).div_ceil(xy);
    assert!(
        z <= max_dim as u64,
        "prefix sum dispatch exceeds device limits"
    );
    [x, y, z as u32]
}
