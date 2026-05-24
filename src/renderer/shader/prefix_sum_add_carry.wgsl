const WG_SIZE: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read> block_sum: array<u32>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

@compute @workgroup_size(WG_SIZE)
fn add_carry(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let data_count = arrayLength(&global_data);
    let block_count = arrayLength(&block_sum);

    let wg_linear = linearize_workgroup_id(wid, num_wg);
    if (wg_linear >= block_count) {
        return;
    }

    let global_idx = wg_linear * WG_SIZE + lid.x;
    if (global_idx >= data_count) {
        return;
    }

    let carry_seed = select(0u, block_sum[wg_linear], sg_lane == 0u);
    let carry = subgroupBroadcastFirst(carry_seed);
    global_data[global_idx] += carry;
}
