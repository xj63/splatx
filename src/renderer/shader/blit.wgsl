@group(0) @binding(0)
var source_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = textureDimensions(source_texture);
    let max_coord = vec2<i32>(max(vec2<u32>(1u), size) - vec2<u32>(1u));
    let texel = vec2<i32>(clamp(in.uv * vec2<f32>(size), vec2<f32>(0.0), vec2<f32>(max_coord)));
    return textureLoad(source_texture, texel, 0);
}
