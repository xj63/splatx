const WG_SIZE: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sum: array<u32>;

var<workgroup> local_data: array<u32, 128u>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn block_scan(lid_x: u32, value: u32) {
    local_data[lid_x] = value;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if (offset >= WG_SIZE) {
            break;
        }

        let current = local_data[lid_x];
        var add = 0u;
        if (lid_x >= offset) {
            add = local_data[lid_x - offset];
        }
        workgroupBarrier();
        local_data[lid_x] = current + add;
        workgroupBarrier();

        offset = offset * 2u;
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_write_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let n = arrayLength(&global_data);
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;
    let value = select(0u, global_data[global_idx], in_range);

    block_scan(lid.x, value);

    if (lid.x == WG_SIZE - 1u) {
        let block_count = arrayLength(&block_sum);
        if (wg_linear < block_count) {
            block_sum[wg_linear] = local_data[lid.x];
        }
    }
    workgroupBarrier();

    if (in_range) {
        global_data[global_idx] = select(0u, local_data[lid.x - 1u], lid.x > 0u);
    }
}

@compute @workgroup_size(WG_SIZE)
fn block_scan_no_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let n = arrayLength(&global_data);
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let global_idx = wg_linear * WG_SIZE + lid.x;
    let in_range = global_idx < n;
    let value = select(0u, global_data[global_idx], in_range);

    block_scan(lid.x, value);

    if (in_range) {
        global_data[global_idx] = select(0u, local_data[lid.x - 1u], lid.x > 0u);
    }
}
