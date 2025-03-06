#include <metal_stdlib>
#include "params.metal"

using namespace metal;

inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

kernel void FFNSwiGLU_Step1(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],
    const device float *w1         [[ buffer(2) ]],
    const device float *w3         [[ buffer(3) ]],
    device float *res              [[ buffer(4) ]],
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].actual_hidden_dim) {
        return;
    }
    
    float val1 = 0.0;
    float val3 = 0.0;
    uint base = b * param[0].seq_len + i;
    uint input_ptr = base * param[0].dim;
    uint res_ptr = base * param[0].actual_hidden_dim;
    uint w_ptr = j * param[0].dim;
    for (uint k = 0; k < param[0].dim; k++, input_ptr++, w_ptr++) {
        val1 += input[input_ptr] * w1[w_ptr];
        val3 += input[input_ptr] * w3[w_ptr];
    }
    res[res_ptr + j] = val1 * val3 * sigmoid(val1);
}

kernel void FFNSwiGLU_Step2(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *residual   [[ buffer(1) ]],
    const device float *w2         [[ buffer(2) ]],
    const device float *interres   [[ buffer(3) ]],
    device float *output           [[ buffer(4) ]],
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }
    
    float val = 0.0;
    uint base = b * param[0].seq_len + i;
    uint interres_ptr = base * param[0].actual_hidden_dim;
    uint w_ptr = j * param[0].actual_hidden_dim;
    uint output_ptr = base * param[0].dim + j;

    for (uint k = 0; k < param[0].actual_hidden_dim; k++, interres_ptr++, w_ptr++) {
        val += interres[interres_ptr] * w2[w_ptr];
    }
    if (param[0].residual > 0) {
        output[output_ptr] = val + residual[output_ptr];
    } else {
        output[output_ptr] = val;
    }
}
