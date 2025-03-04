#include <metal_stdlib>

using namespace metal;

inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

kernel void Transformer_TokenEmbedding(
    constant int *params           [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen]
    const device float *emb_table  [[ buffer(2) ]],     // [vocab, dim]
    device float *input_emb        [[ buffer(3) ]],     // [batch, seqlen, dim]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint seqlen = params[1];
    uint dim = params[3];

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= batch || i >= seqlen || j >= dim) {
        return;
    }
    
    int res_ptr = (b * seqlen + i) * dim + j;
    int input_ptr = b * seqlen + i;
    int emb_ptr = int(input[input_ptr]) * dim + j;
    input_emb[res_ptr] = emb_table[emb_ptr];
}


kernel void Transformer_Result(
    constant int *params           [[ buffer(0) ]],
    const device float *input      [[ buffer(1) ]],     // [batch, seqlen, dim]
    const device float *wo         [[ buffer(2) ]],     // [vocab, dim]
    device float *output           [[ buffer(3) ]],     // [batch, seqlen, vocab]
    uint3 gid                      [[ thread_position_in_grid ]])
{
    uint batch = params[0];
    uint seqlen = params[1];
    uint dim = params[3];
    uint vocab_size = params[5];

    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;
    if (b >= batch || i >= seqlen || j >= vocab_size) {
        return;
    }
    
    int res_ptr = (b * seqlen + i) * vocab_size + j;
    int input_ptr = (b * seqlen + i) * dim;
    int wo_ptr = j * dim;
    float val = 0;
    for (int k = 0; k < dim; k++, input_ptr++, wo_ptr++) {
        val += input[input_ptr] * wo[wo_ptr];
    }
    output[res_ptr] = val;
}

