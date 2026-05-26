use std::{
    path::{Path, PathBuf},
    sync::mpsc,
};

use clap::Parser;
use glam::Vec3;
use image::{ImageBuffer, Rgba};
use splatx::{
    camera::Camera,
    model::SplatxModel,
    renderer::{RenderTarget, Renderer, recommended_device_features},
};

#[derive(Debug, Parser)]
#[command(
    name = "render-image",
    about = "Render a splatx model to a PNG using a preset camera."
)]
struct Args {
    /// Input splatx NPZ model path.
    model: PathBuf,

    /// Output PNG path.
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,

    /// Time parameter passed to the renderer.
    #[arg(short, long, default_value_t = 0.0)]
    time: f32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct PresetCamera {
    camera: Camera,
    width: u32,
    height: u32,
}

fn preset_camera() -> PresetCamera {
    PresetCamera {
        camera: Camera {
            position: Vec3::new(0.44396468, -1.1035035, -0.3499273),
            target: Vec3::new(0.54787546, -1.0966363, 0.6446356),
            // The roll is encoded in `up`; `Camera` itself stays a plain look-at camera.
            up: Vec3::new(-0.02173217, 0.9997531, -0.004632412),
            fovy_radians: 1.2135941,
            znear: 8.831384,
            zfar: 109.77542,
        },
        width: 2704,
        height: 2028,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logger();

    let args = Args::parse();
    let model = load_model(&args.model)?;
    let preset = preset_camera();
    let camera = preset.camera;
    let (width, height) = (preset.width, preset.height);

    tracing::info!(
        model = %args.model.display(),
        width,
        height,
        time = args.time,
        "rendering image with preset camera"
    );

    let gpu = pollster::block_on(create_gpu())?;
    let mut renderer = Renderer::new(&gpu.device, &gpu.queue, model);
    let pixels = render_image(&gpu, &mut renderer, &camera, args.time, width, height)?;
    write_png(&args.output, width, height, pixels)?;

    tracing::info!(output = %args.output.display(), "wrote image");
    Ok(())
}

fn load_model(path: &Path) -> Result<SplatxModel, Box<dyn std::error::Error + Send + Sync>> {
    tracing::info!(model = %path.display(), "loading model");
    SplatxModel::load_npz(path)
}

async fn create_gpu() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|error| error.to_string())?;

    let required_features = recommended_device_features(&adapter);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("splatx render-image device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| error.to_string())?;

    Ok(GpuContext { device, queue })
}

fn render_image(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    camera: &Camera,
    time: f32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("splatx render-image texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = width * 4;
    let padded_bytes_per_row = align_to(bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback_size = padded_bytes_per_row as u64 * height as u64;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("splatx render-image readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("splatx render-image encoder"),
        });

    renderer.render(
        camera,
        time,
        RenderTarget {
            encoder: &mut encoder,
            queue: &gpu.queue,
            color_view: &view,
            format,
            width,
            height,
        },
    );
    renderer.analyze_visibility_buffers(&gpu.device, &mut encoder, time);

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.submit([encoder.finish()]);

    let (sender, receiver) = mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let mapped = readback.get_mapped_range(..);
    let mut pixels = vec![0_u8; (bytes_per_row * height) as usize];
    for row in 0..height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * bytes_per_row as usize;
        let src = &mapped[src_offset..src_offset + bytes_per_row as usize];
        let dst = &mut pixels[dst_offset..dst_offset + bytes_per_row as usize];
        dst.copy_from_slice(src);
    }
    drop(mapped);
    readback.unmap();

    Ok(pixels)
}

fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, pixels)
        .ok_or("failed to create image buffer")?;
    image.save(path)?;
    Ok(())
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn init_logger() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
}
