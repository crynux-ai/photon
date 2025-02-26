#pragma once

#include "include/backend.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>

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
