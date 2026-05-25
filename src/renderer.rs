mod alive_count;
mod compact;
mod cull;
mod data;
mod indirect;
mod prefix_sum;
mod profiler;
mod project;
mod sort;
mod util;

use crate::{camera::Camera, model::SplatxModel};

use self::{
    alive_count::AliveCountStage,
    compact::CompactStage,
    cull::{CullParams, CullStage},
    data::{GpuModelData, upload_model},
    indirect::IndirectStage,
    prefix_sum::PrefixSumStage,
    profiler::GpuProfiler,
    project::{PROJECT_WORKGROUP_SIZE, ProjectStage},
    sort::SortStage,
    util::{schedule_depth_sort_validation_log, schedule_u32_buffer_stats_log},
};

pub struct RenderTarget<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub queue: &'a wgpu::Queue,
    pub color_view: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl RenderTarget<'_> {
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

pub struct Renderer {
    model: SplatxModel,
    data: GpuModelData,
    cull: CullStage,
    prefix_sum: PrefixSumStage,
    compact: CompactStage,
    indirect: IndirectStage,
    project: ProjectStage,
    sort: SortStage,
    alive_count: AliveCountStage,
    profiler: GpuProfiler,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, model: SplatxModel) -> Self {
        let data = upload_model(device, &model);
        let len = model.len() as u32;
        let profiler = GpuProfiler::new(device, queue);
        let cull = CullStage::new(device, &data, len);
        let prefix_sum = PrefixSumStage::new(device, len);
        let compact = CompactStage::new(device, len, cull.mask(), prefix_sum.prefix());
        let indirect = IndirectStage::new(
            device,
            len,
            PROJECT_WORKGROUP_SIZE,
            cull.mask(),
            prefix_sum.prefix(),
        );
        let project = ProjectStage::new(
            device,
            len,
            &data,
            compact.alive_indices(),
            indirect.dispatch_args(),
        );
        let sort = SortStage::new(
            device,
            len,
            project.depths(),
            compact.alive_indices(),
            indirect.dispatch_args(),
        );
        let alive_count = AliveCountStage::new(device, len);

        Self {
            model,
            data,
            cull,
            prefix_sum,
            compact,
            indirect,
            project,
            sort,
            alive_count,
            profiler,
        }
    }

    pub fn render(&mut self, camera: &Camera, time: f32, target: RenderTarget<'_>) {
        let mut target = target;
        let view_projection = camera
            .view_projection_matrix(target.aspect_ratio())
            .to_cols_array();

        let mut profiler = self.profiler.begin_frame(time);

        Self::clear(&mut target, wgpu::Color::BLACK);

        self.cull.execute(
            target.encoder,
            target.queue,
            &mut profiler,
            CullParams {
                view_projection,
                time,
            },
        );
        self.prefix_sum
            .execute(target.encoder, &mut profiler, self.cull.mask());
        self.compact
            .execute(target.encoder, target.queue, &mut profiler);
        self.alive_count.execute(
            target.encoder,
            self.cull.mask(),
            self.prefix_sum.prefix(),
            time,
        );
        self.indirect.execute(target.encoder, &mut profiler);
        self.project.execute(
            target.encoder,
            target.queue,
            &mut profiler,
            camera,
            time,
            target.width,
            target.height,
        );
        self.sort.execute(target.encoder, &mut profiler);

        profiler.finish(target.encoder);
    }

    /// Development-only helper that schedules readback and logs statistics for
    /// visibility-related GPU buffers. This is not part of the normal render path.
    pub fn analyze_visibility_buffers(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        time: f32,
    ) {
        let len = self.model.len();
        schedule_u32_buffer_stats_log(device, encoder, "mask", self.cull.mask(), len);
        schedule_u32_buffer_stats_log(device, encoder, "prefix", self.prefix_sum.prefix(), len);
        schedule_u32_buffer_stats_log(
            device,
            encoder,
            "alive_indices",
            self.compact.alive_indices(),
            len,
        );
        schedule_u32_buffer_stats_log(
            device,
            encoder,
            "indirect",
            self.indirect.dispatch_args(),
            4,
        );
        schedule_u32_buffer_stats_log(
            device,
            encoder,
            "sorted_indices",
            self.sort.sorted_indices(),
            len,
        );
        schedule_depth_sort_validation_log(
            device,
            encoder,
            self.sort.sorted_keys(),
            self.sort.sorted_indices(),
            self.indirect.dispatch_args(),
            len,
        );
    }

    pub fn clear(target: &mut RenderTarget<'_>, color: wgpu::Color) {
        target
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splatx clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
    }
}

pub fn recommended_device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let supported = adapter.features();

    let profiler_need =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let prefix_sum_need = wgpu::Features::SUBGROUP;

    supported & (profiler_need | prefix_sum_need)
}
