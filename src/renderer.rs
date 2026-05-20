use crate::{camera::Camera, model::SplatxModel};

pub struct RenderTarget<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
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
}

impl Renderer {
    pub fn new(model: SplatxModel) -> Self {
        Self { model }
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        t: f32,
        target: RenderTarget<'_>,
    ) -> Result<(), String> {
        let _model = &self.model;
        let _view_projection = camera.view_projection_matrix(target.aspect_ratio());
        let _t = t;

        Self::clear(target, wgpu::Color::BLACK);

        // 4DGS render implementation will be added here.
        Ok(())
    }

    pub fn clear(target: RenderTarget<'_>, color: wgpu::Color) {
        let _pass = target
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

use crate::model::F16Array;
use half::f16;
use ndarray::ArrayD;
fn print_array(name: &str, array: &F16Array) {
    let stats = stats(array);
    tracing::info!(
        "{name}: shape={:?} dtype=f16 len={} finite={} mean={:.9} var={:.9} min={:.9} max={:.9}",
        array.shape(),
        stats.len,
        stats.finite,
        stats.mean,
        stats.variance,
        stats.min,
        stats.max,
    );
}

fn stats(array: &ArrayD<f16>) -> Stats {
    let mut len = 0_usize;
    let mut finite = 0_usize;
    let mut mean = 0_f64;
    let mut m2 = 0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for value in array.iter().map(|value| value.to_f32()) {
        len += 1;
        if !value.is_finite() {
            continue;
        }

        finite += 1;
        min = min.min(value);
        max = max.max(value);

        let value = value as f64;
        let delta = value - mean;
        mean += delta / finite as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = if finite > 0 {
        m2 / finite as f64
    } else {
        f64::NAN
    };
    if finite == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }

    Stats {
        len,
        finite,
        mean,
        variance,
        min,
        max,
    }
}

struct Stats {
    len: usize,
    finite: usize,
    mean: f64,
    variance: f64,
    min: f32,
    max: f32,
}
