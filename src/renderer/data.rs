use glam::{Mat3, Quat, Vec3};
use half::f16;
use wgpu::util::DeviceExt;

use crate::model::SplatxModel;

pub struct GpuModelData {
    pub gaussians: wgpu::Buffer,
    pub covariances: wgpu::Buffer,
    pub features_static: wgpu::Buffer,
    pub features_view: wgpu::Buffer,
}

pub fn upload_model(device: &wgpu::Device, model: &SplatxModel) -> GpuModelData {
    GpuModelData {
        gaussians: upload_f16_storage(device, "splatx gaussians", &build_gaussians(model)),
        covariances: upload_f16_storage(device, "splatx covariances", &build_covariances(model)),
        features_static: upload_f16_storage(
            device,
            "splatx static features",
            &build_padded_features(&model.features_static),
        ),
        features_view: upload_f16_storage(
            device,
            "splatx view features",
            &build_padded_features(&model.features_view),
        ),
    }
}

fn build_gaussians(model: &SplatxModel) -> Vec<f16> {
    let mut output = Vec::with_capacity(model.len() * 8);

    for index in 0..model.len() {
        output.extend_from_slice(&model.means[index]);
        output.push(model.times[index]);
        output.extend_from_slice(&model.velocities[index]);
        output.push(f16::from_f32(model.durations[index].to_f32().exp()));
    }

    output
}

fn build_covariances(model: &SplatxModel) -> Vec<f16> {
    let mut output = Vec::with_capacity(model.len() * 6);

    for index in 0..model.len() {
        let covariance = covariance_from_scale_rotation(model.scales[index], model.quats[index]);
        output.extend(covariance.into_iter().map(f16::from_f32));
    }

    output
}

fn build_padded_features(features: &[[f16; 3]]) -> Vec<f16> {
    let mut output = Vec::with_capacity(features.len() * 4);

    for feature in features {
        output.extend_from_slice(feature);
        output.push(f16::ZERO);
    }

    output
}

fn covariance_from_scale_rotation(scale: [f16; 3], quat: [f16; 4]) -> [f32; 6] {
    let rotation = normalized_rotation(quat);
    let sx = scale[0].to_f32().exp();
    let sy = scale[1].to_f32().exp();
    let sz = scale[2].to_f32().exp();
    let covariance =
        rotation * Mat3::from_diagonal(Vec3::new(sx * sx, sy * sy, sz * sz)) * rotation.transpose();

    [
        covariance.x_axis.x,
        covariance.y_axis.x,
        covariance.z_axis.x,
        covariance.y_axis.y,
        covariance.z_axis.y,
        covariance.z_axis.z,
    ]
}

fn normalized_rotation([w, x, y, z]: [f16; 4]) -> Mat3 {
    let quat = Quat::from_xyzw(x.to_f32(), y.to_f32(), z.to_f32(), w.to_f32());
    if quat.length_squared() > 0.0 {
        Mat3::from_quat(quat.normalize())
    } else {
        Mat3::IDENTITY
    }
}

fn upload_f16_storage(device: &wgpu::Device, label: &str, data: &[f16]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}
