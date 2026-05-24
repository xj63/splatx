use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, OffscreenCanvas};

use crate::{
    camera::Camera,
    model::SplatxModel,
    renderer::{RenderTarget, Renderer, recommended_device_features},
};

#[wasm_bindgen(start)]
pub fn start() {
    init_logger();
}

fn init_logger() {
    // we can call the function at least once during initialization,
    // and then we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    console_error_panic_hook::set_once();

    tracing_wasm::set_as_global_default();
}

#[wasm_bindgen]
pub struct WebRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    camera: Camera,
    renderer: Option<Renderer>,
}

impl WebRenderer {
    async fn create_from_target(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<WebRenderer, String> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(target)
            .map_err(|err| err.to_string())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|err| err.to_string())?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("splatx web device"),
                required_features: recommended_device_features(&adapter),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|err| err.to_string())?;

        let config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .ok_or_else(|| "surface is not supported by the selected adapter".to_string())?;
        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            camera: Camera::default(),
            renderer: None,
        })
    }
}

#[wasm_bindgen]
impl WebRenderer {
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        let width = canvas.width();
        let height = canvas.height();
        Self::create_from_target(wgpu::SurfaceTarget::Canvas(canvas), width, height)
            .await
            .map_err(|err| JsValue::from_str(&err))
    }

    pub async fn create_offscreen(canvas: OffscreenCanvas) -> Result<WebRenderer, JsValue> {
        let width = canvas.width();
        let height = canvas.height();
        Self::create_from_target(wgpu::SurfaceTarget::OffscreenCanvas(canvas), width, height)
            .await
            .map_err(|err| JsValue::from_str(&err))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn load_npz_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let cursor = std::io::Cursor::new(bytes);
        let model = SplatxModel::load_npz_reader(cursor)
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        self.renderer = Some(Renderer::new(&self.device, &self.queue, model));
        Ok(())
    }

    pub fn set_camera(
        &mut self,
        position_x: f32,
        position_y: f32,
        position_z: f32,
        target_x: f32,
        target_y: f32,
        target_z: f32,
        up_x: f32,
        up_y: f32,
        up_z: f32,
        fovy_radians: f32,
        znear: f32,
        zfar: f32,
    ) {
        self.camera = Camera {
            position: glam::Vec3::new(position_x, position_y, position_z),
            target: glam::Vec3::new(target_x, target_y, target_z),
            up: glam::Vec3::new(up_x, up_y, up_z),
            fovy_radians,
            znear,
            zfar,
        };
    }

    pub fn render(&mut self, t: f32) -> Result<(), JsValue> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(JsValue::from_str("failed to acquire surface texture"));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splatx web encoder"),
            });

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.render(
                &self.camera,
                t,
                RenderTarget {
                    encoder: &mut encoder,
                    queue: &self.queue,
                    color_view: &view,
                    format: self.config.format,
                    width: self.config.width,
                    height: self.config.height,
                },
            )
        }

        self.queue.submit([encoder.finish()]);
        frame.present();

        Ok(())
    }
}
