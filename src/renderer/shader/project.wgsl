struct ProjectUniform {
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    focal_time_kernel: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: ProjectUniform;

// word0 = mean.x, mean.y
// word1 = mean.z, time
// word2 = velocity.x, velocity.y
// word3 = velocity.z, duration
@group(0) @binding(1)
var<storage, read> gaussians: array<u32>;

// word0 = cov.xx, cov.xy
// word1 = cov.xz, cov.yy
// word2 = cov.yz, cov.zz
@group(0) @binding(2)
var<storage, read> covariances: array<u32>;

@group(0) @binding(3)
var<storage, read> alive_indices: array<u32>;

// values = [dispatch_x, dispatch_y, dispatch_z, alive_count]
@group(0) @binding(4)
var<storage, read> dispatch_args: array<u32>;

@group(0) @binding(5)
var<storage, read> rgba: array<vec4<f32>>;

@group(0) @binding(6)
var<storage, read_write> projected_splats: array<vec4<f32>>;

@group(0) @binding(7)
var<storage, read_write> depths: array<f32>;

fn load_position(index: u32) -> vec3<f32> {
    let base = index * 4u;
    let mean_xy = unpack2x16float(gaussians[base]);
    let mean_zt = unpack2x16float(gaussians[base + 1u]);
    let velocity_xy = unpack2x16float(gaussians[base + 2u]);
    let velocity_zd = unpack2x16float(gaussians[base + 3u]);
    let mean = vec3<f32>(mean_xy.x, mean_xy.y, mean_zt.x);
    let t0 = mean_zt.y;
    let velocity = vec3<f32>(velocity_xy.x, velocity_xy.y, velocity_zd.x);
    return mean + (uniforms.focal_time_kernel.z - t0) * velocity;
}

fn load_covariance(index: u32) -> mat3x3<f32> {
    let base = index * 3u;
    let cov0 = unpack2x16float(covariances[base]);
    let cov1 = unpack2x16float(covariances[base + 1u]);
    let cov2 = unpack2x16float(covariances[base + 2u]);

    return mat3x3<f32>(
        vec3<f32>(cov0.x, cov0.y, cov1.x),
        vec3<f32>(cov0.y, cov1.y, cov2.x),
        vec3<f32>(cov1.x, cov2.x, cov2.y),
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let alive_slot = id.x;
    if (alive_slot >= dispatch_args[3u]) {
        return;
    }
    let gaussian_index = alive_indices[alive_slot];
    let world_position = load_position(gaussian_index);
    let world_covariance = load_covariance(gaussian_index);

    let camera_position4 = uniforms.view * vec4<f32>(world_position, 1.0);
    let camera_z = max(camera_position4.z, 1e-4);
    let camera_x = camera_position4.x;
    let camera_y = camera_position4.y;

    let pos2d = uniforms.view_projection * vec4<f32>(world_position, 1.0);
    let v_center = pos2d / pos2d.w;

    let J = mat3x3<f32>(
        vec3<f32>(uniforms.focal_time_kernel.x / camera_z, 0.0, 0.0),
        vec3<f32>(0.0, uniforms.focal_time_kernel.y / camera_z, 0.0),
        vec3<f32>(
            -uniforms.focal_time_kernel.x * camera_x / (camera_z * camera_z),
            -uniforms.focal_time_kernel.y * camera_y / (camera_z * camera_z),
            0.0,
        ),
    );
    let W = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );
    let projected_covariance = J * W * world_covariance * transpose(W) * transpose(J);

    let cov00 = projected_covariance[0][0];
    let cov01 = projected_covariance[1][0];
    let cov11 = projected_covariance[1][1];
    let kernel = uniforms.focal_time_kernel.w;

    /// according to Mip-Splatting by Yu et al. 2023
    let det_0 = max(1e-6, cov00 * cov11 - cov01 * cov01);
    let det_1 = max(1e-6, (cov00 + kernel) * (cov11 + kernel) - cov01 * cov01);
    var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);
    if (det_0 <= 1e-6 || det_1 <= 1e-6) {
        coef = 0.0;
    }

    let diagonal1 = cov00 + kernel;
    let off_diagonal = cov01;
    let diagonal2 = cov11 + kernel;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) * 0.5, off_diagonal));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let direction = normalize(vec2<f32>(off_diagonal, lambda1 - diagonal1));
    let v1 = sqrt(2.0 * lambda1) * direction;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(direction.y, -direction.x);

    let final_rgba = vec4<f32>(rgba[alive_slot].xyz, rgba[alive_slot].w * coef);
    projected_splats[alive_slot * 3u] = vec4<f32>(v_center.xy, camera_z, final_rgba.w);
    projected_splats[alive_slot * 3u + 1u] = vec4<f32>(v1, v2);
    projected_splats[alive_slot * 3u + 2u] = final_rgba;
    depths[alive_slot] = camera_z;
}
