struct IndirectUniform {
    values: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: IndirectUniform;

@group(0) @binding(1)
var<storage, read> mask: array<u32>;

@group(0) @binding(2)
var<storage, read> prefix: array<u32>;

// values = [dispatch_x, dispatch_y, dispatch_z, alive_count]
@group(0) @binding(3)
var<storage, read_write> dispatch_args: array<u32>;

fn ceil_div(value: u32, divisor: u32) -> u32 {
    return (value + divisor - 1u) / divisor;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x > 0u || uniforms.values.x == 0u) {
        dispatch_args[0] = 0u;
        dispatch_args[1] = 1u;
        dispatch_args[2] = 1u;
        dispatch_args[3] = 0u;
        return;
    }

    let last = uniforms.values.x - 1u;
    let alive_count = prefix[last] + mask[last];

    dispatch_args[0] = ceil_div(alive_count, uniforms.values.y);
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
    dispatch_args[3] = alive_count;
}
