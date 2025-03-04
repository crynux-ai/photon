#pragma once

#include "include/backend.h"
#include "include/executor.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>

enum FFNSwiGLUTensor {
    FFNSwiGLU_INPUT = 0,
    FFNSwiGLU_PARAM_1 = 1,
    FFNSwiGLU_W1 = 2,
    FFNSwiGLU_W2 = 3,
    FFNSwiGLU_W3 = 4,
    FFNSwiGLU_PARAM_2 = 5,
    FFNSwiGLU_RESIDUAL = 6,
    FFNSwiGLU_RESULT = 7,
    FFNSwiGLU_HIDDEN_OUTPUT = 8,
};


template <BackendType backend>
class FFNSwiGLU {

public:
    virtual ~FFNSwiGLU() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual Tensor forward(const Tensor& input, Tensor* residual=nullptr) = 0;

protected:
    FFNSwiGLU(int dim, int hidden_dim, int multiple_of) = default;
};

template <> class FFNSwiGLU<BackendType::CPU>;

template <> class FFNSwiGLU<BackendType::METAL>;


#ifdef PHOTON_METAL
#include "metal/feedforward.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/feedforward.h"
    #endif
#endif
