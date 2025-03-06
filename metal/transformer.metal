#include <metal_stdlib>
#include "params.metal"

using namespace metal;

inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

kernel void Transformer_TokenEmbedding(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen]
    const device float *emb_table  [[ buffer(2) ]],     // [vocab, dim]
    device float *input_emb        [[ buffer(3) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].dim) {
        return;
    }
    
    int res_ptr = (b * param[0].seq_len + i) * param[0].dim + j;
    int input_ptr = b * param[0].seq_len + i;
    int emb_ptr = int(input[input_ptr]) * param[0].dim + j;
    input_emb[res_ptr] = emb_table[emb_ptr];
}


kernel void Transformer_Result(
    constant RunParams *param      [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *wo         [[ buffer(2) ]],     // [vocab, dim]
    device float *output           [[ buffer(3) ]],     // [batch, seqlen, vocab]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= param[0].batch || i >= param[0].seq_len || j >= param[0].vocab_size) {
        return;
    }
    
    int res_ptr = (b * param[0].seq_len + i) * param[0].vocab_size + j;
    int input_ptr = (b * param[0].seq_len + i) * param[0].dim;
    int wo_ptr = j * param[0].dim;
    float val = 0;
    for (int k = 0; k < param[0].dim; k++, input_ptr++, wo_ptr++) {
        val += input[input_ptr] * wo[wo_ptr];
    }
    output[res_ptr] = val;
}

