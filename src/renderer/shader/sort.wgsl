struct SortUniform {
    values: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SortUniform;

// values = [dispatch_x, dispatch_y, dispatch_z, alive_count]
@group(0) @binding(1)
var<storage, read> dispatch_args: array<u32>;

@group(0) @binding(2)
var<storage, read> depths: array<f32>;

@group(0) @binding(3)
var<storage, read> alive_indices: array<u32>;

@group(0) @binding(4)
var<storage, read_write> histograms: array<u32>;

@group(0) @binding(5)
var<storage, read_write> keys_a: array<u32>;

@group(0) @binding(6)
var<storage, read_write> keys_b: array<u32>;

@group(0) @binding(7)
var<storage, read_write> payload_a: array<u32>;

@group(0) @binding(8)
var<storage, read_write> payload_b: array<u32>;

const RADIX_SIZE: u32 = 256u;

fn alive_count() -> u32 {
    return dispatch_args[3u];
}

fn block_size() -> u32 {
    return uniforms.values.z;
}

fn pass_index() -> u32 {
    return uniforms.values.w;
}

fn num_blocks() -> u32 {
    return uniforms.values.y;
}

fn encode_depth_key(depth: f32) -> u32 {
    return 0xffffffffu - bitcast<u32>(depth);
}

fn digit_of(key: u32, pass_index: u32) -> u32 {
    return (key >> (pass_index * 8u)) & 0xffu;
}

@compute @workgroup_size(64)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= alive_count()) {
        return;
    }

    keys_a[index] = encode_depth_key(depths[index]);
    payload_a[index] = alive_indices[index];
}

@compute @workgroup_size(1)
fn histogram(@builtin(workgroup_id) wid: vec3<u32>) {
    let block = wid.x;
    let base = block * block_size();
    let histogram_base = block * RADIX_SIZE;

    for (var digit = 0u; digit < RADIX_SIZE; digit = digit + 1u) {
        histograms[histogram_base + digit] = 0u;
    }

    let limit = min(base + block_size(), alive_count());
    for (var index = base; index < limit; index = index + 1u) {
        let key = select(keys_b[index], keys_a[index], pass_index() % 2u == 0u);
        let digit = digit_of(key, pass_index());
        histograms[histogram_base + digit] = histograms[histogram_base + digit] + 1u;
    }
}

@compute @workgroup_size(1)
fn prefix_histograms(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    var global_offset = 0u;
    for (var digit = 0u; digit < RADIX_SIZE; digit = digit + 1u) {
        var digit_offset = global_offset;
        for (var block = 0u; block < num_blocks(); block = block + 1u) {
            let offset = block * RADIX_SIZE + digit;
            let count = histograms[offset];
            histograms[offset] = digit_offset;
            digit_offset = digit_offset + count;
        }
        global_offset = digit_offset;
    }
}

fn scatter_one_pass(pass_index: u32, write_to_b: bool, block: u32) {
    let base = block * block_size();
    let limit = min(base + block_size(), alive_count());
    var local_counts: array<u32, 256>;

    for (var digit = 0u; digit < RADIX_SIZE; digit = digit + 1u) {
        local_counts[digit] = 0u;
    }

    for (var index = base; index < limit; index = index + 1u) {
        let key = select(keys_b[index], keys_a[index], write_to_b);
        let payload = select(payload_b[index], payload_a[index], write_to_b);
        let digit = digit_of(key, pass_index);
        let target_index = histograms[block * RADIX_SIZE + digit] + local_counts[digit];
        local_counts[digit] = local_counts[digit] + 1u;

        if (write_to_b) {
            keys_b[target_index] = key;
            payload_b[target_index] = payload;
        } else {
            keys_a[target_index] = key;
            payload_a[target_index] = payload;
        }
    }
}

@compute @workgroup_size(1)
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>) {
    scatter_one_pass(pass_index(), true, wid.x);
}

@compute @workgroup_size(1)
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>) {
    scatter_one_pass(pass_index(), false, wid.x);
}
