pub fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn schedule_u32_buffer_stats_log(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    buffer: &wgpu::Buffer,
    len: usize,
) {
    if len == 0 {
        tracing::info!(buffer = label, len = 0, "buffer stats");
        return;
    }

    let byte_len = (len * std::mem::size_of::<u32>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("splatx buffer stats readback"),
        size: byte_len.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, byte_len);

    let mapped = readback.clone();
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, 0..byte_len, move |result| {
        if let Err(error) = result {
            tracing::warn!(buffer = label, "failed to read buffer stats: {error}");
            return;
        }

        let bytes = mapped.get_mapped_range(..byte_len);
        let values = bytemuck::cast_slice::<u8, u32>(&bytes);
        let stats = u32_stats(values);
        tracing::info!(
            buffer = label,
            len = stats.len,
            mean = stats.mean,
            variance = stats.variance,
            min = stats.min,
            max = stats.max,
            first3 = %format_u32_slice(&stats.first),
            last3 = %format_u32_slice(&stats.last),
            "buffer stats"
        );
        drop(bytes);
        mapped.unmap();
    });
}

struct U32Stats {
    len: usize,
    mean: f64,
    variance: f64,
    min: u32,
    max: u32,
    first: Vec<u32>,
    last: Vec<u32>,
}

fn u32_stats(values: &[u32]) -> U32Stats {
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut min = u32::MAX;
    let mut max = u32::MIN;

    for (index, &value) in values.iter().enumerate() {
        min = min.min(value);
        max = max.max(value);

        let value_f64 = value as f64;
        let delta = value_f64 - mean;
        mean += delta / (index + 1) as f64;
        let delta2 = value_f64 - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / values.len() as f64;
    let first = values.iter().take(3).copied().collect::<Vec<_>>();
    let mut last = values.iter().rev().take(3).copied().collect::<Vec<_>>();
    last.reverse();

    U32Stats {
        len: values.len(),
        mean,
        variance,
        min,
        max,
        first,
        last,
    }
}

fn format_u32_slice(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}
