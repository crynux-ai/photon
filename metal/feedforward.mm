#include "metal/feedforward.h"

#include <cassert>


void FFNSwiGLU<BackendType::METAL>::forward(int seqlen, bool residual) {
    int batch = _executor->batch;
    size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
    size_t wByteSize = _dim * _hidden_dim * sizeof(float);
    size_t resByteSize = batch * seqlen * _hidden_dim * sizeof(float);

    Tensor hidden_output({batch, seqlen, _hidden_dim});
    Tensor result({batch, seqlen, _dim});

    // silu(w1(x)) * w3(x)
    int params[] = {batch, seqlen, _dim, _hidden_dim};
    _executor->addBuffer(obj_id, FFNSwiGLU_PARAM_1, params, 4*sizeof(int));
    _executor->addBuffer(obj_id, FFNSwiGLU_HIDDEN_OUTPUT, hidden_output._value.get(), resByteSize);

    _executor->forward(obj_id, 1,
        {
            FFNSwiGLU_PARAM_1,
            FFNSwiGLU_INPUT,
            FFNSwiGLU_W1,
            FFNSwiGLU_W3,
            FFNSwiGLU_HIDDEN_OUTPUT,
        },
        {batch, seqlen, _hidden_dim});

    // w2(res)
    int params2[] = {batch, seqlen, _dim, _hidden_dim, residual ? 1 : 0};
    _executor->addBuffer(obj_id, FFNSwiGLU_PARAM_2, params2, 5*sizeof(int));
    _executor->addBuffer(obj_id, FFNSwiGLU_RESULT, result._value.get(), inputByteSize);
    if (!residual) {
        _executor->addBuffer(obj_id, FFNSwiGLU_RESIDUAL, obj_id, FFNSwiGLU_INPUT);  // dummy input
    }

    _executor->forward(obj_id, 2,
        {
            FFNSwiGLU_PARAM_2,
            FFNSwiGLU_RESIDUAL,
            FFNSwiGLU_W2,
            FFNSwiGLU_HIDDEN_OUTPUT,
            FFNSwiGLU_RESULT,
        },
        {batch, seqlen, _dim});

    _executor->bufferToTensor(obj_id, FFNSwiGLU_RESULT, &result); 
}
