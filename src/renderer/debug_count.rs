pub struct DebugCountStage {
    readback: wgpu::Buffer,
    len: u32,
}

impl DebugCountStage {
    pub fn new(device: &wgpu::Device, len: u32) -> Self {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx debug-count readback"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { readback, len }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mask: &wgpu::Buffer,
        prefix: &wgpu::Buffer,
    ) {
        if self.len == 0 {
            return;
        }

        let last_offset = (self.len as u64 - 1) * 4;
        encoder.copy_buffer_to_buffer(prefix, last_offset, &self.readback, 0, 4);
        encoder.copy_buffer_to_buffer(mask, last_offset, &self.readback, 4, 4);

        let readback = self.readback.clone();
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, ..8, move |result| {
            if let Err(error) = result {
                tracing::warn!("failed to read debug cull count: {error}");
                return;
            }

            let bytes = readback.get_mapped_range(..8);
            let prefix_last = u32::from_le_bytes(bytes[0..4].try_into().expect("prefix last"));
            let mask_last = u32::from_le_bytes(bytes[4..8].try_into().expect("mask last"));
            let alive_gaussians = prefix_last + mask_last;
            tracing::info!(alive_gaussians, "temporary prefix-sum debug");
            drop(bytes);
            readback.unmap();
        });
    }
}
