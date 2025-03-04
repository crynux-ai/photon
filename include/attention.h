#pragma once

#include "include/backend.h"
#include "include/rope.h"
#include "schema/tensor.h"
#include <cmath>

enum AttentionTensor {
    Attention_INPUT = 0,
    Attention_CACHE_K = 1,
    Attention_CACHE_V = 2,
    Attention_WEIGHT_Q = 3,
    Attention_WEIGHT_K = 4,
    Attention_WEIGHT_V = 5,
    Attention_WEIGHT_O = 6,
    Attention_XQ = 7,
    Attention_X_PARAMS = 8,
    Attention_ROPE_PARAMS = 9,
    Attention_ROPE_COST = 10,
    Attention_ROPE_SINT = 11,
    Attention_SCORE_PARAMS = 12,
    Attention_SCORE = 13,
    Attention_OUTPUT = 14,
    Attention_RESULT = 15,
    Attention_RESULT_PARAMS = 16,
    Attention_RESIDUAL = 17,
};

template <BackendType backend>
class Attention {

public:
    virtual ~Attention() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual Tensor forward(
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual=nullptr) = 0;

protected:
    Attention(int dim, int num_heads) = default;
};

template <> class Attention<BackendType::CPU>;

template <> class Attention<BackendType::METAL>;

#ifdef PHOTON_METAL
#include "metal/attention.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/attention.h"
    #endif
#endif
