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


    @autoreleasepool {
        // wq(input), wk(input), wv(input)
        int params[] = {batch, seqlen, _max_seq_len, _dim, start_pos};
        _executor->addBuffer(obj_id, AttentionTensor::INPUT, input._value.get(), inputByteSize);
        _executor->addBuffer(obj_id, AttentionTensor::XQ, xq._value.get(), inputByteSize);
        _executor->addBuffer(obj_id, AttentionTensor::X_PARAMS, params, 5*sizeof(int));
        _executor->forward(obj_id, 3,
            {
                AttentionTensor::X_PARAMS,
                AttentionTensor::INPUT,
                AttentionTensor::WEIGHT_Q,
                AttentionTensor::WEIGHT_K,
                AttentionTensor::WEIGHT_V,
                AttentionTensor::XQ,
                AttentionTensor::CACHE_K,
                AttentionTensor::CACHE_V,
            },
            {batch, seqlen, _dim});

        // Rope apply_rotary_emb
        int rope_params[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, num_complex};
        _executor->addBuffer(obj_id, AttentionTensor::ROPE_PARAMS, rope_params, 7*sizeof(float));
        _executor->addBuffer(obj_id, AttentionTensor::ROPE_COST, rope_cost._value.get(), ropeByteSize);
        _executor->addBuffer(obj_id, AttentionTensor::ROPE_SINT, rope_sint._value.get(), ropeByteSize);
        _executor->forward(obj_id, 4,
            {
                AttentionTensor::ROPE_PARAMS,
                AttentionTensor::ROPE_COST,
                AttentionTensor::ROPE_SINT,
                AttentionTensor::XQ,
                AttentionTensor::CACHE_K,
            },
            {batch, seqlen, _dim / 2});


        // softmax(QK^T/scale)) (not averaged)
        int compute_score_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, mask ? 1 : 0};
        _executor->addBuffer(obj_id, AttentionTensor::SCORE_PARAMS, compute_score_param, 7*sizeof(float));
        _executor->addBuffer(obj_id, AttentionTensor::SCORE, score._value.get(), scoreByteSize);
        _executor->forward(obj_id, 5,
            {
                AttentionTensor::SCORE_PARAMS,
                AttentionTensor::XQ,
                AttentionTensor::CACHE_K,
                AttentionTensor::CACHE_V,
                AttentionTensor::SCORE,
            },
            {batch, seqlen, totlen * _num_heads});

        // score @ cachev
        _executor->addBuffer(obj_id, AttentionTensor::OUTPUT, output._value.get(), inputByteSize);
        _executor->forward(obj_id, 6,
            {
                AttentionTensor::SCORE_PARAMS,
                AttentionTensor::SCORE,
                AttentionTensor::CACHE_V,
                AttentionTensor::OUTPUT,
            },
            {batch, seqlen, _dim});

        // Wo @ output
        int result_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, residual ? 1 : 0};
        _executor->addBuffer(obj_id, AttentionTensor::RESULT_PARAMS, result_param, 7*sizeof(float));
        _executor->addBuffer(obj_id, AttentionTensor::RESULT, result._value.get(), inputByteSize);
        int residual_idx = AttentionTensor::INPUT;   // dummy input
        if (residual) {
            _executor->addBuffer(obj_id, AttentionTensor::RESIDUAL, residual->_value.get(), inputByteSize);
            residual_idx = AttentionTensor::RESIDUAL;
        }
        _executor->forward(obj_id, 7,
            {
                AttentionTensor::RESULT_PARAMS,
                AttentionTensor::OUTPUT,
                AttentionTensor::WEIGHT_O,
                residual_idx,
                AttentionTensor::RESULT,
            },
            {batch, seqlen, _dim});
        
        // Convert result
        _executor->bufferToTensor(obj_id, AttentionTensor::RESULT, &result);

        return result;
    }
}
