#include <metal_stdlib>

using namespace metal;

kernel void Attention_Step1(
    constant int *params           [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],
    const device float *wq         [[ buffer(2) ]],
    const device float *wk         [[ buffer(3) ]],
    const device float *wv         [[ buffer(4) ]],
    device float *xq               [[ buffer(5) ]],
    device float *xk               [[ buffer(6) ]],
    device float *xv               [[ buffer(7) ]],
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint seqlen = params[1];
    uint dim = params[2];

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= batch || i >= seqlen || j >= dim) {
        return;
    }
    
    float valq = 0.0;
    float valk = 0.0;
    float valv = 0.0;

    uint base = (b * seqlen + i) * dim;
    uint input_ptr = base;
    uint res_ptr = base + j;
    uint w_ptr = j * dim;
    for (uint k = 0; k < dim; k++, input_ptr++, w_ptr++) {
        valq += input[input_ptr] * wq[w_ptr];
        valk += input[input_ptr] * wk[w_ptr];
        valv += input[input_ptr] * wv[w_ptr];
    }

    xq[res_ptr] = valq;
    xk[res_ptr] = valk;
    xv[res_ptr] = valv;
}
