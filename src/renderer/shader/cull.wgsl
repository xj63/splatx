struct CullUniform {
    view_projection: mat4x4<f32>,
    time: f32,
    gaussian_count: u32,
    _pad0: vec2<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: CullUniform;

// Each gaussian uses 4 u32 words:
// word0 = mean.x, mean.y
// word1 = mean.z, time
// word2 = velocity.x, velocity.y
// word3 = velocity.z, duration
@group(0) @binding(1)
var<storage, read> gaussians: array<u32>;

@group(0) @binding(2)
var<storage, read_write> mask: array<u32>;

fn load_gaussian(index: u32) -> vec4<f32> {
    let base = index * 4u;
    let mean_xy = unpack2x16float(gaussians[base + 0u]);
    let mean_z_time = unpack2x16float(gaussians[base + 1u]);
    let velocity_xy = unpack2x16float(gaussians[base + 2u]);
    let velocity_z_duration = unpack2x16float(gaussians[base + 3u]);

    let mean = vec3<f32>(mean_xy.x, mean_xy.y, mean_z_time.x);
    let t0 = mean_z_time.y;
    let velocity = vec3<f32>(velocity_xy.x, velocity_xy.y, velocity_z_duration.x);
    let duration = velocity_z_duration.y;
    let dt = uniforms.time - t0;

    let temporal_threshold = sqrt(2.0 * log(255.0)) * duration;
    if (abs(dt) >= temporal_threshold) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return vec4<f32>(mean + dt * velocity, 1.0);
}

fn inside_extended_frustum(position: vec4<f32>) -> bool {
    let clip = uniforms.view_projection * position;
    let b = 1.2 * clip.w;

    return clip.w > 0.0
        && -b < clip.x
        && clip.x < b
        && -b < clip.y
        && clip.y < b
        && 0.0 < clip.z
        && clip.z < clip.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= uniforms.gaussian_count) {
        return;
    }

    let position = load_gaussian(index);
    let alive = position.w > 0.0 && inside_extended_frustum(position);
    mask[index] = select(0u, 1u, alive);
}
