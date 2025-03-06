#include <metal_stdlib>
#include "params.metal"

using namespace metal;

kernel void Rope_apply_rotary_emb(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *cost       [[ buffer(1) ]],     // [max_seq_len, num_complex]
    const device float *sint       [[ buffer(2) ]],     // [max_seq_len, num_complex]
    device float *xq               [[ buffer(3) ]],     // [batch, seqlen, dim]
    device float *cachek           [[ buffer(4) ]],     // [batch, max_seq_len, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z * 2;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }

    uint ptr0 = (b * param[0].seq_len + i) * param[0].dim + j;
    uint ptr1 = ptr0 + 1;
    uint ptr_rope = (i + param[0].start_pos) * param[0].num_complex + gid.z % param[0].num_complex;

    float xq0 = xq[ptr0];
    float xq1 = xq[ptr1];
    xq[ptr0] = xq0 * cost[ptr_rope] - xq1 * sint[ptr_rope];
    xq[ptr1] = xq1 * cost[ptr_rope] + xq0 * sint[ptr_rope];

    ptr0 = (b * param[0].max_seq_len + i + param[0].start_pos) * param[0].dim + j;
    ptr1 = ptr0 + 1;
    float xk0 = cachek[ptr0];
    float xk1 = cachek[ptr1];
    cachek[ptr0] = xk0 * cost[ptr_rope] - xk1 * sint[ptr_rope];
    cachek[ptr1] = xk1 * cost[ptr_rope] + xk0 * sint[ptr_rope];
}
