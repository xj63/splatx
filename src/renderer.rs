mod alive_count;
mod appearance;
mod blit;
mod compact;
mod cull;
mod data;
mod indirect;
mod prefix_sum;
mod profiler;
mod project;
mod render;
mod sort;
mod util;

use crate::{camera::Camera, model::SplatxModel};

use self::{
    alive_count::AliveCountStage,
    appearance::AppearanceStage,
    blit::BlitStage,
    compact::CompactStage,
    cull::{CullParams, CullStage},
    data::{GpuModelData, upload_model},
    indirect::IndirectStage,
    prefix_sum::PrefixSumStage,
    profiler::GpuProfiler,
    project::{PROJECT_WORKGROUP_SIZE, ProjectStage},
    render::RenderStage,
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
    device: wgpu::Device,
    model: SplatxModel,
    data: GpuModelData,
    cull: CullStage,
    prefix_sum: PrefixSumStage,
    compact: CompactStage,
    indirect: IndirectStage,
    appearance: AppearanceStage,
    project: ProjectStage,
    sort: SortStage,
    render: RenderStage,
    blit: BlitStage,
    fp16_target: Option<Fp16Target>,
    alive_count: AliveCountStage,
    profiler: GpuProfiler,
}

struct Fp16Target {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
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
        let appearance = AppearanceStage::new(
            device,
            len,
            &data,
            compact.alive_indices(),
            indirect.dispatch_args(),
        );
        let project = ProjectStage::new(
            device,
            len,
            &data.gaussians,
            &data.covariances,
            compact.alive_indices(),
            indirect.dispatch_args(),
            appearance.rgba(),
        );
        let sort = SortStage::new(
            device,
            len,
            project.depths(),
            compact.alive_indices(),
            indirect.dispatch_args(),
        );
        let render = RenderStage::new(
            device,
            project.projected(),
            sort.sorted_indices(),
            indirect.draw_args(),
        );
        let blit = BlitStage::new(device);
        let alive_count = AliveCountStage::new(device, len);

        Self {
            device: device.clone(),
            model,
            data,
            cull,
            prefix_sum,
            compact,
            indirect,
            appearance,
            project,
            sort,
            render,
            blit,
            fp16_target: None,
            alive_count,
            profiler,
        }
    }

    pub fn render(&mut self, camera: &Camera, time: f32, target: RenderTarget<'_>) {
        let mut target = target;
        self.ensure_fp16_target(target.width, target.height);
        let fp16_target = self.fp16_target.as_ref().expect("fp16 target");
        let view_projection = camera
            .view_projection_matrix(target.aspect_ratio())
            .to_cols_array();

        let mut profiler = self.profiler.begin_frame(time);

        Self::clear_view(target.encoder, &fp16_target.view, wgpu::Color::BLACK);

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
        self.appearance
            .execute(target.encoder, target.queue, &mut profiler, camera, time);
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
        self.render.execute(
            &self.device,
            target.queue,
            target.encoder,
            &fp16_target.view,
            wgpu::TextureFormat::Rgba16Float,
            target.width,
            target.height,
        );
        self.blit.execute(
            &self.device,
            target.encoder,
            &fp16_target.view,
            target.color_view,
            target.format,
        );

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

    fn ensure_fp16_target(&mut self, width: u32, height: u32) {
        let needs_recreate = self
            .fp16_target
            .as_ref()
            .map(|target| target.width != width || target.height != height)
            .unwrap_or(true);
        if !needs_recreate {
            return;
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("splatx fp16 render target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.fp16_target = Some(Fp16Target {
            width,
            height,
            texture,
            view,
        });
    }

    fn clear_view(
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        color: wgpu::Color,
    ) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splatx clear pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
