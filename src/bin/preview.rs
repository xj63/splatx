use std::sync::Arc;

use splatx::{
    camera::Camera,
    model::SplatxModel,
    renderer::{RenderTarget, Renderer},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

struct PreviewSurface {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    camera: Camera,
    time: f32,
    renderer: Option<Renderer>,
}

impl PreviewSurface {
    async fn new(
        window: Arc<Window>,
        width: u32,
        height: u32,
        model: Option<SplatxModel>,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
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
                label: Some("splatx preview device"),
                required_features: wgpu::Features::empty(),
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
            time: 0.0,
            renderer: model.map(Renderer::new),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) -> Result<(), String> {
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
                return Err("failed to acquire surface texture".to_string());
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splatx preview encoder"),
            });

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.render(
                &self.camera,
                self.time,
                RenderTarget {
                    encoder: &mut encoder,
                    color_view: &view,
                    format: self.config.format,
                    width: self.config.width,
                    height: self.config.height,
                },
            )?;
        }

        self.queue.submit([encoder.finish()]);
        frame.present();

        Ok(())
    }
}

#[derive(Default)]
struct PreviewApp {
    window: Option<Arc<Window>>,
    surface: Option<PreviewSurface>,
}

impl ApplicationHandler for PreviewApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title("splatx preview"))
                .expect("failed to create preview window"),
        );
        let size = window.inner_size();
        let model = std::env::args()
            .nth(1)
            .map(SplatxModel::load_npz)
            .transpose()
            .expect("failed to load model");
        let surface = pollster::block_on(PreviewSurface::new(
            window.clone(),
            size.width,
            size.height,
            model,
        ))
        .expect("failed to initialize preview surface");

        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(surface) = self.surface.as_mut() {
                    surface.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(surface) = self.surface.as_mut() {
                    surface.render().expect("failed to render frame");
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() -> Result<(), winit::error::EventLoopError> {
    init_logger();

    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut PreviewApp::default())
}

fn init_logger() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();
}
