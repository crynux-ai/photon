#include "include/profiler.h"
#include "metal/feedforward.h"

#include <cassert>


void FFNSwiGLU<BackendType::METAL>::forward(const RunParams& param) {
    // silu(w1(x)) * w3(x)
    _executor->forward(obj_id, 1, param,
        {
            FFNSwiGLU_INPUT,
            FFNSwiGLU_W1,
            FFNSwiGLU_W3,
            FFNSwiGLU_HIDDEN_OUTPUT,
        },
        {param.batch, param.seq_len, param.actual_hidden_dim}
        PROFILE_TAG("FFNSwiGLU/SiluW1*W3"));

    // w2(res) 
    _executor->forward(obj_id, 2, param,
        {
            FFNSwiGLU_RESIDUAL,
            FFNSwiGLU_W2,
            FFNSwiGLU_HIDDEN_OUTPUT,
            FFNSwiGLU_RESULT,
        },
        {param.batch, param.seq_len, _dim}
        PROFILE_TAG("FFNSwiGLU/Output"));
}

void FFNSwiGLU<BackendType::METAL>::build(std::string_view content) {
    PROFILE_BEGIN(obj_id, "FFNSwiGLU/build")
    auto ptr = content.data();
    auto weight_read_size = _dim * _actual_hidden_dim * 4 + 12;

    _w1.build({ptr, static_cast<size_t>(weight_read_size)});
    ptr += weight_read_size;
    _w2.build({ptr, static_cast<size_t>(weight_read_size)});
    ptr += weight_read_size;
    _w3.build({ptr, static_cast<size_t>(weight_read_size)});

    size_t weight_size = _dim * _actual_hidden_dim * sizeof(float);

    _executor->addBuffer(obj_id, FFNSwiGLU_W1, _w1);
    _executor->addBuffer(obj_id, FFNSwiGLU_W2, _w2);
    _executor->addBuffer(obj_id, FFNSwiGLU_W3, _w3);
    PROFILE_END(obj_id, "FFNSwiGLU/build")
}

void FFNSwiGLU<BackendType::METAL>::alloc_shared_buffer(const RunParams& param) {
    PROFILE_BEGIN(obj_id, "FFNSwiGLU/alloc")
    std::vector<int> input_shape = {param.batch, param.seq_len, param.dim};
    _executor->addBuffer(obj_id, FFNSwiGLU_RESULT, input_shape);
    if (!param.residual) {
        _executor->addBuffer(obj_id, FFNSwiGLU_RESIDUAL, obj_id, FFNSwiGLU_INPUT);  // dummy input
    }
    _executor->addBuffer(obj_id, FFNSwiGLU_HIDDEN_OUTPUT, {param.batch, param.seq_len, param.actual_hidden_dim});
    PROFILE_END(obj_id, "FFNSwiGLU/alloc")
}
