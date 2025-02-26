#include <metal_stdlib>

using namespace metal;

inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

kernel void FFNSwiGLU_step1(
    constant int *params           [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],
    const device float *w1         [[ buffer(2) ]],
    const device float *w3         [[ buffer(3) ]],
    device float *res              [[ buffer(4) ]],
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint seqlen = params[1];
    uint dim = params[2];
    uint hidden_dim = params[3];

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= batch || i >= seqlen || j >= hidden_dim) {
        return;
    }
    
    float val1 = 0.0;
    float val3 = 0.0;
    uint base = b * seqlen + i;
    uint input_ptr = base * dim;
    uint res_ptr = base * hidden_dim;
    uint w_ptr = j * dim;
    for (uint k = 0; k < dim; k++, input_ptr++, w_ptr++) {
        val1 += input[input_ptr] * w1[w_ptr];
        val3 += input[input_ptr] * w3[w_ptr];
    }
    res[res_ptr + j] = val1 * val3 * sigmoid(val1);
}
