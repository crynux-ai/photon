#pragma once

#include "include/backend.h"
#include "include/rope.h"
#include "schema/tensor.h"
#include <cmath>

enum AttentionTensor {
    INPUT = 0,
    CACHE_K = 1,
    CACHE_V = 2,
    WEIGHT_Q = 3,
    WEIGHT_K = 4,
    WEIGHT_V = 5,
    WEIGHT_O = 6,
    XQ = 7,
    X_PARAMS = 8,
    ROPE_PARAMS = 9,
    ROPE_COST = 10,
    ROPE_SINT = 11,
    SCORE_PARAMS = 12,
    SCORE = 13,
    OUTPUT = 14,
    RESULT = 15,
    RESULT_PARAMS = 16,
    RESIDUAL = 17,
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
