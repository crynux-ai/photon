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


kernel void Attention_ComputeScore(
    constant int *params           [[ buffer(0) ]],
    const device float *xq         [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *cachek     [[ buffer(2) ]],     // [batch, max_seq_len, dim]
    const device float *cachev     [[ buffer(3) ]],     // [batch, max_seq_len, dim]
    device float *score            [[ buffer(4) ]],     // [batch, seqlen, max_seq_len, num_head]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    int batch = params[0];
    int max_seq_len = params[1];
    int seqlen = params[2];
    int startpos = params[3];
    int dim = params[4];
    int num_heads = params[5];
    int head_dim = dim / num_heads;
    int mask = params[6];
    int totlen = seqlen + startpos;
    float scale = sqrt(float(head_dim));

    int b = gid.x;
    int i = gid.y;
    int j = gid.z;
    if (b >= batch || i >= seqlen || j >= totlen * num_heads) {
        return;
    }
    int h = j % num_heads;
    j = j / num_heads;

    int fill = totlen;
    if (mask > 0) {
        fill = startpos + i;
    }

    int ptr = (b * seqlen + i) * max_seq_len * num_heads + gid.z;
    if (j > fill) {
        score[ptr] = 0;
        return;
    }

    float tmp = 0;
    int ptrq = (b * seqlen + i) * dim + h * head_dim;
    int ptrk = (b * max_seq_len + j) * dim + h * head_dim;
    for (int k = 0; k < head_dim; k++, ptrq++, ptrk++) {
        tmp += xq[ptrq] * cachek[ptrk];
    } 
    score[ptr] = exp(tmp / scale);
}

kernel void Attention_Output(
    constant int *params           [[ buffer(0) ]],
    const device float *score      [[ buffer(1) ]],     // [batch, seqlen, max_seq_len, num_head]
    const device float *cachev     [[ buffer(2) ]],     // [batch, max_seq_len, dim]
    device float *output           [[ buffer(3) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint max_seq_len = params[1];
    uint seqlen = params[2];
    uint startpos = params[3];
    uint dim = params[4];
    uint num_heads = params[5];
    uint head_dim = dim / num_heads;
    uint mask = params[6];
    uint totlen = seqlen + startpos;

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= batch || i >= seqlen || j >= dim) {
        return;
    }
    uint h = j / head_dim;


    uint base = (b * seqlen + i) * max_seq_len * num_heads + h;
    uint ptrscore = base;
    float sum = 0;

    for (int k = 0; k < totlen; k++, ptrscore += num_heads) {
        sum += score[ptrscore];
    }

    ptrscore = base;
    float res = 0;
    uint ptrv = b * max_seq_len * dim + j;
    for (int k = 0; k < totlen; k++, ptrscore += num_heads, ptrv += dim) {
        res += score[ptrscore] * cachev[ptrv];
    }
    output[(b * seqlen + i) * dim + j] = res / sum;
}
