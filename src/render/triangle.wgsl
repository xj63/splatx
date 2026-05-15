struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.65),
        vec2<f32>(-0.65, -0.55),
        vec2<f32>(0.65, -0.55),
    );

    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.25, 0.2),
        vec3<f32>(0.2, 0.85, 0.35),
        vec3<f32>(0.25, 0.45, 1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.color = colors[vertex_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
