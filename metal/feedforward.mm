#include "metal/feedforward.h"

#include <cassert>


Tensor FFNSwiGLU<BackendType::METAL>::forward(const Tensor& input, Tensor* residual) {
    assert(input.shape().size() == 3);
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    assert(input.shape()[2] == _dim);

    size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
    size_t wByteSize = _dim * _hidden_dim * sizeof(float);
    size_t resByteSize = batch * seqlen * _hidden_dim * sizeof(float);

    Tensor hidden_output({batch, seqlen, _hidden_dim});
    Tensor result(input.shape());

    // silu(w1(x)) * w3(x)
    int params[] = {batch, seqlen, _dim, _hidden_dim};
    _executor->addBuffer(obj_id, FFNSwiGLU_PARAM_1, params, 4*sizeof(int));
    _executor->addBuffer(obj_id, FFNSwiGLU_INPUT, input._value.get(), inputByteSize);
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
    int residual_idx = FFNSwiGLU_INPUT;  // Dummy input.
    if (residual) {
        _executor->addBuffer(obj_id, FFNSwiGLU_RESIDUAL, residual->_value.get(), inputByteSize);
        residual_idx = FFNSwiGLU_RESIDUAL;
    }

    _executor->forward(obj_id, 2,
        {
            FFNSwiGLU_PARAM_2,
            residual_idx,
            FFNSwiGLU_W2,
            FFNSwiGLU_HIDDEN_OUTPUT,
            FFNSwiGLU_RESULT,
        },
        {batch, seqlen, _dim});

    _executor->bufferToTensor(obj_id, FFNSwiGLU_RESULT, &result);
    return result;
    
}
