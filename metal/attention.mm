#include "metal/attention.h"

#include <cassert>


Tensor Attention<BackendType::METAL>::forward(
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual) {
    int batch = _executor->batch;
    assert(batch == input.shape()[0]);
    int seqlen = input.shape()[1];
    int num_complex = rope_cost.shape()[1];
    int totlen = start_pos + seqlen;

    size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
    size_t wByteSize = _dim * _dim * sizeof(float);
    size_t qByteSize = batch * seqlen * _dim * sizeof(float);
    size_t ropeByteSize = _max_seq_len * num_complex * sizeof(float);
    int scoreByteSize = batch * seqlen * _max_seq_len * _num_heads * sizeof(float);

    Tensor xq({batch, seqlen, _dim});
    Tensor score({batch, seqlen, _max_seq_len, _num_heads});
    Tensor output({batch, seqlen, _dim});
    Tensor result({batch, seqlen, _dim});


    // wq(input), wk(input), wv(input)
    int params[] = {batch, seqlen, _max_seq_len, _dim, start_pos};
    _executor->addBuffer(obj_id, Attention_INPUT, input._value.get(), inputByteSize);
    _executor->addBuffer(obj_id, Attention_XQ, xq._value.get(), inputByteSize);
    _executor->addBuffer(obj_id, Attention_X_PARAMS, params, 5*sizeof(int));
    _executor->forward(obj_id, 3,
        {
            Attention_X_PARAMS,
            Attention_INPUT,
            Attention_WEIGHT_Q,
            Attention_WEIGHT_K,
            Attention_WEIGHT_V,
            Attention_XQ,
            Attention_CACHE_K,
            Attention_CACHE_V,
        },
        {batch, seqlen, _dim});

    // Rope apply_rotary_emb
    int rope_params[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, num_complex};
    _executor->addBuffer(obj_id, Attention_ROPE_PARAMS, rope_params, 7*sizeof(float));
    _executor->addBuffer(obj_id, Attention_ROPE_COST, rope_cost._value.get(), ropeByteSize);
    _executor->addBuffer(obj_id, Attention_ROPE_SINT, rope_sint._value.get(), ropeByteSize);
    _executor->forward(obj_id, 4,
        {
            Attention_ROPE_PARAMS,
            Attention_ROPE_COST,
            Attention_ROPE_SINT,
            Attention_XQ,
            Attention_CACHE_K,
        },
        {batch, seqlen, _dim / 2});


    // softmax(QK^T/scale)) (not averaged)
    int compute_score_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, mask ? 1 : 0};
    _executor->addBuffer(obj_id, Attention_SCORE_PARAMS, compute_score_param, 7*sizeof(float));
    _executor->addBuffer(obj_id, Attention_SCORE, score._value.get(), scoreByteSize);
    _executor->forward(obj_id, 5,
        {
            Attention_SCORE_PARAMS,
            Attention_XQ,
            Attention_CACHE_K,
            Attention_CACHE_V,
            Attention_SCORE,
        },
        {batch, seqlen, totlen * _num_heads});

    // score @ cachev
    _executor->addBuffer(obj_id, Attention_OUTPUT, output._value.get(), inputByteSize);
    _executor->forward(obj_id, 6,
        {
            Attention_SCORE_PARAMS,
            Attention_SCORE,
            Attention_CACHE_V,
            Attention_OUTPUT,
        },
        {batch, seqlen, _dim});

    // Wo @ output
    int result_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, residual ? 1 : 0};
    _executor->addBuffer(obj_id, Attention_RESULT_PARAMS, result_param, 7*sizeof(float));
    _executor->addBuffer(obj_id, Attention_RESULT, result._value.get(), inputByteSize);
    int residual_idx = Attention_INPUT;   // dummy input
    if (residual) {
        _executor->addBuffer(obj_id, Attention_RESIDUAL, residual->_value.get(), inputByteSize);
        residual_idx = Attention_RESIDUAL;
    }
    _executor->forward(obj_id, 7,
        {
            Attention_RESULT_PARAMS,
            Attention_OUTPUT,
            Attention_WEIGHT_O,
            residual_idx,
            Attention_RESULT,
        },
        {batch, seqlen, _dim});
    
    // Convert result
    _executor->bufferToTensor(obj_id, Attention_RESULT, &result);

    return result;
}
