struct CompactUniform {
    values: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: CompactUniform;

@group(0) @binding(1)
var<storage, read> mask: array<u32>;

@group(0) @binding(2)
var<storage, read> prefix: array<u32>;

@group(0) @binding(3)
var<storage, read_write> alive_indices: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= uniforms.values.x) {
        return;
    }

    if (mask[index] == 0u) {
        return;
    }

    let alive_index = prefix[index];
    alive_indices[alive_index] = index;
}
