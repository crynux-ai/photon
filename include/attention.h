#pragma once

#include "include/backend.h"
#include "layers/rope.h"
#include "schema/tensor.h"
#include <cmath>


template <BackendType backend>
class Attention {

public:
    virtual ~Attention() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual Tensor forward(
        const Tensor& input, const std::pair<FreqMatrix, FreqMatrix>& rope,
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
