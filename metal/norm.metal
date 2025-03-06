#include <metal_stdlib>
#include "params.metal"

using namespace metal;

kernel void Norm_RMS(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen, dim]
    device float *output           [[ buffer(2) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }

    uint ptr = (b * param[0].seq_len + i) * param[0].dim + j;
    output[ptr] = input[ptr] * input[ptr];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = param[0].dim / 2; i > 0; i /= 2) {
        if (ptr < i) {
            output[ptr] += output[ptr + i];
        } else {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f / (sqrt(output[0] / param[0].dim) + param[0].norm_eps);
    output[ptr] = input[ptr] * scale;
}
