#include "metal/feedforward.h"

#include <cassert>


void FFNSwiGLU<BackendType::METAL>::forward(const RunParams& param) {
    Tensor hidden_output({param.batch, param.seq_len, param.actual_hidden_dim});
    Tensor result({param.batch, param.seq_len, _dim});

    // silu(w1(x)) * w3(x)
    _executor->addBuffer(obj_id, FFNSwiGLU_HIDDEN_OUTPUT, hidden_output);
    _executor->forward(obj_id, 1, param,
        {
            FFNSwiGLU_INPUT,
            FFNSwiGLU_W1,
            FFNSwiGLU_W3,
            FFNSwiGLU_HIDDEN_OUTPUT,
        },
        {param.batch, param.seq_len, param.actual_hidden_dim});

    // w2(res)
    _executor->addBuffer(obj_id, FFNSwiGLU_RESULT, result);
    if (!param.residual) {
        _executor->addBuffer(obj_id, FFNSwiGLU_RESIDUAL, obj_id, FFNSwiGLU_INPUT);  // dummy input
    }
    _executor->forward(obj_id, 2, param,
        {
            FFNSwiGLU_RESIDUAL,
            FFNSwiGLU_W2,
            FFNSwiGLU_HIDDEN_OUTPUT,
            FFNSwiGLU_RESULT,
        },
        {param.batch, param.seq_len, _dim});
}
