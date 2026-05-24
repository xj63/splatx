const MAX_PROFILED_DISPATCHES: usize = 32;
const QUERY_COUNT: usize = MAX_PROFILED_DISPATCHES * 2;
const TIMESTAMP_BYTES: usize = QUERY_COUNT * 8;

pub struct GpuProfiler {
    backend: Option<TimestampBackend>,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let need_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;

        let backend = if device.features().contains(need_features) {
            Some(TimestampBackend::new(device, queue))
        } else {
            tracing::info!("timestamp query is unavailable; gpu profiling is disabled");
            None
        };

        Self { backend }
    }

    pub fn begin_frame(&mut self) -> GpuProfilerFrame<'_> {
        GpuProfilerFrame {
            backend: self.backend.as_mut(),
            stage_names: Vec::with_capacity(MAX_PROFILED_DISPATCHES),
            next_query: 0,
        }
    }
}

pub struct GpuProfilerFrame<'a> {
    backend: Option<&'a mut TimestampBackend>,
    stage_names: Vec<&'static str>,
    next_query: u32,
}

impl GpuProfilerFrame<'_> {
    pub fn profile_compute(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        label: &'static str,
        stage: impl FnOnce(&mut wgpu::ComputePass<'_>),
    ) {
        if let Some((query_set, start, end)) = self.reserve_queries(label) {
            pass.write_timestamp(query_set, start);
            stage(pass);
            pass.write_timestamp(query_set, end);
        } else {
            stage(pass);
        }
    }

    pub fn finish(self, encoder: &mut wgpu::CommandEncoder) {
        let Some(backend) = self.backend else {
            return;
        };
        if self.stage_names.is_empty() {
            return;
        }

        let byte_len = self.next_query as u64 * 8;
        encoder.resolve_query_set(&backend.query_set, 0..self.next_query, &backend.resolve, 0);
        encoder.copy_buffer_to_buffer(&backend.resolve, 0, &backend.readback, 0, byte_len);

        let readback = backend.readback.clone();
        let frame = CompletedFrame {
            stage_names: self.stage_names,
            period_ns: backend.period_ns,
        };

        encoder.map_buffer_on_submit(
            &backend.readback,
            wgpu::MapMode::Read,
            0..byte_len,
            move |result| {
                if let Err(error) = result {
                    tracing::warn!("failed to read gpu profiler timestamps: {error}");
                    return;
                }

                let view = readback.get_mapped_range(..byte_len);
                frame.log(&view);
                drop(view);
                readback.unmap();
            },
        );
    }

    fn reserve_queries(&mut self, label: &'static str) -> Option<(&wgpu::QuerySet, u32, u32)> {
        if self.stage_names.len() >= MAX_PROFILED_DISPATCHES {
            return None;
        }
        self.stage_names.push(label);

        let backend = self.backend.as_deref_mut()?;
        let start = self.next_query;
        let end = start + 1;
        self.next_query += 2;
        Some((&backend.query_set, start, end))
    }
}

struct CompletedFrame {
    stage_names: Vec<&'static str>,
    period_ns: f32,
}

impl CompletedFrame {
    fn log(&self, bytes: &[u8]) {
        for (index, stage) in self.stage_names.iter().enumerate() {
            let start_offset = index * 16;
            let end_offset = start_offset + 8;
            let start = u64::from_le_bytes(
                bytes[start_offset..start_offset + 8]
                    .try_into()
                    .expect("dispatch start timestamp"),
            );
            let end = u64::from_le_bytes(
                bytes[end_offset..end_offset + 8]
                    .try_into()
                    .expect("dispatch end timestamp"),
            );
            let elapsed_ms = end.saturating_sub(start) as f64 * self.period_ns as f64 / 1_000_000.0;
            tracing::info!(stage, elapsed_ms, "gpu dispatch");
        }
    }
}

struct TimestampBackend {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
}

impl TimestampBackend {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("splatx dispatch profiler query set"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT as u32,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx dispatch profiler resolve"),
            size: TIMESTAMP_BYTES as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatx dispatch profiler readback"),
            size: TIMESTAMP_BYTES as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve,
            readback,
            period_ns: queue.get_timestamp_period(),
        }
    }
}
