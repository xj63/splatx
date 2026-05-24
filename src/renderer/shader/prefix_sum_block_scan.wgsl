const WG_SIZE: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sum: array<u32>;

var<workgroup> local_data: array<u32, 128u>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_write_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
) {
    let n = arrayLength(&global_data);
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;

    var value = 0u;
    if (in_range) {
        value = global_data[global_idx];
    }

    let subgroup_prefix = subgroupExclusiveAdd(value);
    let subgroup_sum = subgroupAdd(value);

    if (sg_lane == 0u) {
        local_data[sg_id] = subgroup_sum;
    }
    workgroupBarrier();

    let subgroup_count = (WG_SIZE + sg_size - 1u) / sg_size;
    if (lid.x == 0u) {
        var total = 0u;
        for (var i = 0u; i < subgroup_count; i = i + 1u) {
            let current = local_data[i];
            local_data[i] = total;
            total = total + current;
        }

        let block_count = arrayLength(&block_sum);
        if (wg_linear < block_count) {
            block_sum[wg_linear] = total;
        }
    }
    workgroupBarrier();

    if (in_range) {
        global_data[global_idx] = local_data[sg_id] + subgroup_prefix;
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_no_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_id) sg_id: u32,
) {
    let n = arrayLength(&global_data);
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;

    var value = 0u;
    if (in_range) {
        value = global_data[global_idx];
    }

    let subgroup_prefix = subgroupExclusiveAdd(value);
    let subgroup_sum = subgroupAdd(value);

    if (sg_lane == 0u) {
        local_data[sg_id] = subgroup_sum;
    }
    workgroupBarrier();

    let subgroup_count = (WG_SIZE + sg_size - 1u) / sg_size;
    if (lid.x == 0u) {
        var total = 0u;
        for (var i = 0u; i < subgroup_count; i = i + 1u) {
            let current = local_data[i];
            local_data[i] = total;
            total = total + current;
        }
    }
    workgroupBarrier();

    if (in_range) {
        global_data[global_idx] = local_data[sg_id] + subgroup_prefix;
    }
}
