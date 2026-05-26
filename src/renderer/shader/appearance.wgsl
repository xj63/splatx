const PI: f32 = 3.14159265359;
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2: array<f32, 5> = array<f32, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
);
const SH_C3: array<f32, 7> = array<f32, 7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
);

const MLP_CONT_LAYER1_OFFSET: u32 = 0u;
const MLP_CONT_LAYER2_OFFSET: u32 = 96u * 64u;

const MLP_DC_BASE: u32 = 0u;
const MLP_OPACITY_BASE: u32 = 2048u;
const MLP_SH_BASE: u32 = 4096u;

struct AppearanceUniform {
    camera_position: vec4<f32>,
    params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: AppearanceUniform;

// word0 = mean.x, mean.y
// word1 = mean.z, time
// word2 = velocity.x, velocity.y
// word3 = velocity.z, duration
@group(0) @binding(1)
var<storage, read> gaussians: array<u32>;

@group(0) @binding(2)
var<storage, read> features_static: array<u32>;

@group(0) @binding(3)
var<storage, read> features_view: array<u32>;

@group(0) @binding(4)
var<storage, read> weights_cont: array<u32>;

@group(0) @binding(5)
var<storage, read> weights_attr: array<u32>;

@group(0) @binding(6)
var<storage, read> alive_indices: array<u32>;

// values = [dispatch_x, dispatch_y, dispatch_z, alive_count]
@group(0) @binding(7)
var<storage, read> dispatch_args: array<u32>;

@group(0) @binding(8)
var<storage, read_write> rgba: array<vec4<f32>>;

fn relu(x: f32) -> f32 {
    return max(x, 0.0);
}

fn leaky_relu(x: f32) -> f32 {
    return select(0.01 * x, x, x > 0.0);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn attr_weight(scalar_index: u32) -> f32 {
    let pair = unpack2x16float(weights_attr[scalar_index / 2u]);
    return select(pair.x, pair.y, (scalar_index % 2u) == 1u);
}

fn cont_weight(scalar_index: u32) -> f32 {
    let pair = unpack2x16float(weights_cont[scalar_index / 2u]);
    return select(pair.x, pair.y, (scalar_index % 2u) == 1u);
}

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
    let dt = uniforms.params.x - t0;
    let mean_t = mean + dt * velocity;
    let temporal_opacity = exp(-0.5 * (dt / duration) * (dt / duration));

    return vec4<f32>(mean_t, temporal_opacity);
}

fn contract_to_unisphere(pos: vec3<f32>) -> vec3<f32> {
    let x = pos;
    let mag = length(x);
    if (mag > 1.0) {
        return ((2.0 - 1.0 / mag) * (x / mag)) / 4.0 + 0.5;
    }
    return x / 4.0 + 0.5;
}

fn load_static_feature3(index: u32) -> vec3<f32> {
    let base = index * 2u;
    let pair0 = unpack2x16float(features_static[base + 0u]);
    let pair1 = unpack2x16float(features_static[base + 1u]);
    return vec3<f32>(pair0.x, pair0.y, pair1.x);
}

fn load_view_feature3(index: u32) -> vec3<f32> {
    let base = index * 2u;
    let pair0 = unpack2x16float(features_view[base + 0u]);
    let pair1 = unpack2x16float(features_view[base + 1u]);
    return vec3<f32>(pair0.x, pair0.y, pair1.x);
}

fn sh_rest(hidden64: array<f32, 64>, coef_index: u32) -> vec3<f32> {
    let output_base = MLP_SH_BASE + 16u * 64u + coef_index * 3u * 64u;
    var result = vec3<f32>(0.0);

    for (var channel: u32 = 0u; channel < 3u; channel = channel + 1u) {
        var sum = 0.0;
        let row_base = output_base + channel * 64u;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            sum += hidden64[i] * attr_weight(row_base + i);
        }
        if (channel == 0u) {
            result.x = sum;
        } else if (channel == 1u) {
            result.y = sum;
        } else {
            result.z = sum;
        }
    }

    return result;
}

fn evaluate_sh_degree3(dir: vec3<f32>, dc: vec3<f32>, hidden64: array<f32, 64>) -> vec3<f32> {
    var result = SH_C0 * dc;

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    result += -SH_C1 * y * sh_rest(hidden64, 0u);
    result += SH_C1 * z * sh_rest(hidden64, 1u);
    result += -SH_C1 * x * sh_rest(hidden64, 2u);

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    result += SH_C2[0] * xy * sh_rest(hidden64, 3u);
    result += SH_C2[1] * yz * sh_rest(hidden64, 4u);
    result += SH_C2[2] * (2.0 * zz - xx - yy) * sh_rest(hidden64, 5u);
    result += SH_C2[3] * xz * sh_rest(hidden64, 6u);
    result += SH_C2[4] * (xx - yy) * sh_rest(hidden64, 7u);

    result += SH_C3[0] * y * (3.0 * xx - yy) * sh_rest(hidden64, 8u);
    result += SH_C3[1] * xy * z * sh_rest(hidden64, 9u);
    result += SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_rest(hidden64, 10u);
    result += SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rest(hidden64, 11u);
    result += SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_rest(hidden64, 12u);
    result += SH_C3[5] * z * (xx - yy) * sh_rest(hidden64, 13u);
    result += SH_C3[6] * x * (xx - 3.0 * yy) * sh_rest(hidden64, 14u);

    return max(result + 0.5, vec3<f32>(0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let alive_slot = id.x;
    if (alive_slot >= dispatch_args[3u]) {
        return;
    }

    let gaussian_index = alive_indices[alive_slot];
    let gaussian = load_gaussian(gaussian_index);
    let contracted = contract_to_unisphere(gaussian.xyz);

    var encoded: array<f32, 96>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let freq = pow(2.0, f32(i)) * PI;

        let vx = contracted.x * freq;
        encoded[i * 2u + 0u] = sin(vx);
        encoded[i * 2u + 1u] = cos(vx);

        let vy = contracted.y * freq;
        encoded[32u + i * 2u + 0u] = sin(vy);
        encoded[32u + i * 2u + 1u] = cos(vy);

        let vz = contracted.z * freq;
        encoded[64u + i * 2u + 0u] = sin(vz);
        encoded[64u + i * 2u + 1u] = cos(vz);
    }

    var hidden64: array<f32, 64>;
    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
        var sum = 0.0;
        for (var i: u32 = 0u; i < 96u; i = i + 1u) {
            sum += encoded[i] * cont_weight(MLP_CONT_LAYER1_OFFSET + j * 96u + i);
        }
        hidden64[j] = relu(sum);
    }

    var cont_feature: array<f32, 13>;
    for (var j: u32 = 0u; j < 13u; j = j + 1u) {
        var sum = 0.0;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            sum += hidden64[i] * cont_weight(MLP_CONT_LAYER2_OFFSET + j * 64u + i);
        }
        cont_feature[j] = sum;
    }

    let static_feature = load_static_feature3(gaussian_index);
    let view_feature = load_view_feature3(gaussian_index);

    var space_latent: array<f32, 16>;
    var view_latent: array<f32, 16>;
    for (var i: u32 = 0u; i < 13u; i = i + 1u) {
        space_latent[i] = cont_feature[i];
        view_latent[i] = cont_feature[i];
    }
    space_latent[13] = static_feature.x;
    space_latent[14] = static_feature.y;
    space_latent[15] = static_feature.z;
    view_latent[13] = view_feature.x;
    view_latent[14] = view_feature.y;
    view_latent[15] = view_feature.z;

    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
        var sum = 0.0;
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            sum += space_latent[i] * attr_weight(MLP_DC_BASE + j * 16u + i);
        }
        hidden64[j] = leaky_relu(sum);
    }

    var dc = vec3<f32>(0.0);
    let dc_output_base = MLP_DC_BASE + 16u * 64u;
    for (var channel: u32 = 0u; channel < 3u; channel = channel + 1u) {
        var sum = 0.0;
        let row_base = dc_output_base + channel * 64u;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            sum += hidden64[i] * attr_weight(row_base + i);
        }
        if (channel == 0u) {
            dc.x = sum;
        } else if (channel == 1u) {
            dc.y = sum;
        } else {
            dc.z = sum;
        }
    }

    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
        var sum = 0.0;
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            sum += space_latent[i] * attr_weight(MLP_OPACITY_BASE + j * 16u + i);
        }
        hidden64[j] = leaky_relu(sum);
    }

    var opacity_sum = 0.0;
    let opacity_output_base = MLP_OPACITY_BASE + 16u * 64u;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        opacity_sum += hidden64[i] * attr_weight(opacity_output_base + i);
    }
    let opacity = sigmoid(opacity_sum) * gaussian.w;

    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
        var sum = 0.0;
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            sum += view_latent[i] * attr_weight(MLP_SH_BASE + j * 16u + i);
        }
        hidden64[j] = leaky_relu(sum);
    }

    let dir = normalize(gaussian.xyz - uniforms.camera_position.xyz);
    let color = evaluate_sh_degree3(dir, dc, hidden64);
    rgba[alive_slot] = vec4<f32>(color, opacity);
}
