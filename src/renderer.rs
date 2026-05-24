mod cull;
mod data;
mod debug_count;
mod prefix_sum;
mod profiler;
mod util;

use crate::{camera::Camera, model::SplatxModel};

use self::{
    cull::{CullParams, CullStage},
    data::{GpuModelData, upload_model},
    debug_count::DebugCountStage,
    prefix_sum::PrefixSumStage,
    profiler::{GpuProfiler, GpuProfilerFrame},
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
    debug_count: DebugCountStage,
    profiler: GpuProfiler,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: SplatxModel,
    ) -> Result<Self, String> {
        let data = upload_model(device, &model);
        let len = model.len() as u32;
        let profiler = GpuProfiler::new(device, queue);
        let cull = CullStage::new(device, &data, len);
        let prefix_sum = PrefixSumStage::new(device, len)?;
        let debug_count = DebugCountStage::new(device, len);

        Ok(Self {
            model,
            data,
            cull,
            prefix_sum,
            debug_count,
            profiler,
        })
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        time: f32,
        target: RenderTarget<'_>,
    ) -> Result<(), String> {
        let mut target = target;
        let view_projection = camera
            .view_projection_matrix(target.aspect_ratio())
            .to_cols_array();

        let mut profiler = self.profiler.begin_frame();

        Self::clear(&mut target, wgpu::Color::BLACK);

        self.cull.execute(
            target.encoder,
            target.queue,
            &mut profiler,
            CullParams {
                view_projection,
                time,
            },
        )?;
        self.prefix_sum
            .execute(target.encoder, &mut profiler, self.cull.mask());
        self.debug_count
            .execute(target.encoder, self.cull.mask(), self.prefix_sum.prefix());

        profiler.finish(target.encoder);
        Ok(())
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
