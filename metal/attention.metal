#include <metal_stdlib>

using namespace metal;

kernel void Attention_Step1(
    constant int *params           [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *wq         [[ buffer(2) ]],     // [dim, dim]
    const device float *wk         [[ buffer(3) ]],     // [dim, dim]
    const device float *wv         [[ buffer(4) ]],     // [dim, dim]
    device float *xq               [[ buffer(5) ]],     // [batch, seqlen, dim]
    device float *cachek           [[ buffer(6) ]],     // [batch, max_seq_len, dim]
    device float *cachev           [[ buffer(7) ]],     // [batch, max_seq_len, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint seqlen = params[1];
    uint max_seq_len = params[2];
    uint dim = params[3];
    uint startpos = params[4];

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
    uint w_ptr = j * dim;
    for (uint k = 0; k < dim; k++, input_ptr++, w_ptr++) {
        valq += input[input_ptr] * wq[w_ptr];
        valk += input[input_ptr] * wk[w_ptr];
        valv += input[input_ptr] * wv[w_ptr];
    }

    uint res_ptr = base + j;
    xq[res_ptr] = valq;

    res_ptr = (b * max_seq_len + i + startpos) * dim + j;
    cachek[res_ptr] = valk;
    cachev[res_ptr] = valv;
}
