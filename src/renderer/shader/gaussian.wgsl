const CUTOFF: f32 = 2.3539888583335364;

struct RenderUniform {
    viewport: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: RenderUniform;

@group(0) @binding(1)
var<storage, read> projected_splats: array<vec4<f32>>;

@group(1) @binding(0)
var<storage, read> indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let slot = indices[in_instance_index];
    let center_depth_alpha = projected_splats[slot * 3u];
    let basis = projected_splats[slot * 3u + 1u];
    let color = projected_splats[slot * 3u + 2u];

    // filp-y
    // let v1 = basis.xy;
    // let v2 = basis.zw;
    // let v_center = center_depth_alpha.xy;
    let v1 = vec2<f32>(basis.x, -basis.y);
    let v2 = vec2<f32>(basis.z, -basis.w);
    let v_center = vec2<f32>(center_depth_alpha.x, -center_depth_alpha.y);

    let x = select(-1.0, 1.0, in_vertex_index % 2u == 0u);
    let y = select(-1.0, 1.0, in_vertex_index < 2u);
    let screen_pos = vec2<f32>(x, y) * CUTOFF;

    let viewport = uniforms.viewport.xy;
    let offset = (2.0 * mat2x2<f32>(v1 / viewport, v2 / viewport)) * screen_pos;

    out.position = vec4<f32>(v_center + offset, 0.0, 1.0);
    out.screen_pos = screen_pos;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if (a > 2.0 * CUTOFF) {
        discard;
    }

    let b = min(0.99, exp(-a) * in.color.a);
    return vec4<f32>(in.color.rgb, 1.0) * b;
}
