#include <metal_stdlib>
#include "params.metal"

using namespace metal;

kernel void Attention_Step1(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *wq         [[ buffer(2) ]],     // [dim, dim]
    const device float *wk         [[ buffer(3) ]],     // [dim, dim]
    const device float *wv         [[ buffer(4) ]],     // [dim, dim]
    device float *xq               [[ buffer(5) ]],     // [batch, seqlen, dim]
    device float *cachek           [[ buffer(6) ]],     // [batch, max_seq_len, dim]
    device float *cachev           [[ buffer(7) ]],     // [batch, max_seq_len, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }
    
    float valq = 0.0;
    float valk = 0.0;
    float valv = 0.0;

    uint base = (b * param[0].seq_len + i) * param[0].dim;
    uint input_ptr = base;
    uint w_ptr = j * param[0].dim;
    for (uint k = 0; k < param[0].dim; k++, input_ptr++, w_ptr++) {
        valq += input[input_ptr] * wq[w_ptr];
        valk += input[input_ptr] * wk[w_ptr];
        valv += input[input_ptr] * wv[w_ptr];
    }

    uint res_ptr = base + j;
    xq[res_ptr] = valq;

    res_ptr = (b * param[0].max_seq_len + i + param[0].start_pos) * param[0].dim + j;
    cachek[res_ptr] = valk;
    cachev[res_ptr] = valv;
}


kernel void Attention_ComputeScore(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *xq         [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *cachek     [[ buffer(2) ]],     // [batch, max_seq_len, dim]
    const device float *cachev     [[ buffer(3) ]],     // [batch, max_seq_len, dim]
    device float *score            [[ buffer(4) ]],     // [batch, seqlen, max_seq_len, num_head]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    int totlen = param[0].seq_len + param[0].start_pos;
    float scale = sqrt(float(param[0].head_dim));

    int b = gid.x;
    int i = gid.y;
    int j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= totlen * param[0].num_heads) {
        return;
    }
    int h = j % param[0].num_heads;
    j = j / param[0].num_heads;

    int fill = totlen;
    if (param[0].mask > 0) {
        fill = param[0].start_pos + i;
    }

    int ptr = (b * param[0].seq_len + i) * param[0].max_seq_len * param[0].num_heads + gid.z;
    if (j > fill) {
        score[ptr] = 0;
        return;
    }

    float tmp = 0;
    int ptrq = (b * param[0].seq_len + i) * param[0].dim + h * param[0].head_dim;
    int ptrk = (b * param[0].max_seq_len + j) * param[0].dim + h * param[0].head_dim;
    for (int k = 0; k < param[0].head_dim; k++, ptrq++, ptrk++) {
        tmp += xq[ptrq] * cachek[ptrk];
    } 
    score[ptr] = exp(tmp / scale);
}

kernel void Attention_Output(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *score      [[ buffer(1) ]],     // [batch, seqlen, max_seq_len, num_head]
    const device float *cachev     [[ buffer(2) ]],     // [batch, max_seq_len, dim]
    device float *output           [[ buffer(3) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint totlen = param[0].seq_len + param[0].start_pos;

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }
    uint h = j / param[0].head_dim;

    uint base = (b * param[0].seq_len + i) * param[0].max_seq_len * param[0].num_heads + h;
    uint ptrscore = base;
    float sum = 0;

    for (int k = 0; k < totlen; k++, ptrscore += param[0].num_heads) {
        sum += score[ptrscore];
    }

    ptrscore = base;
    float res = 0;
    uint ptrv = b * param[0].max_seq_len * param[0].dim + j;
    for (int k = 0; k < totlen; k++, ptrscore += param[0].num_heads, ptrv += param[0].dim) {
        res += score[ptrscore] * cachev[ptrv];
    }
    output[(b * param[0].seq_len + i) * param[0].dim + j] = res / sum;
}

kernel void Attention_Result(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *output     [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *wo         [[ buffer(2) ]],     // [batch, dim, dim]
    const device float *residual   [[ buffer(3) ]],     // [batch, seqlen, dim]
    device float *result           [[ buffer(4) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }
    
    float tmp = 0;
    int ptrw = j * param[0].dim;
    int ptr = (b * param[0].seq_len + i) * param[0].dim;
    int ptro = ptr;
    for (int k = 0; k < param[0].dim; k++, ptrw++, ptro++) {
        tmp += wo[ptrw] * output[ptro];
    }

    ptr += j;
    if (param[0].residual > 0) {
        tmp += residual[ptr];
    }
    result[ptr] = tmp;
}

