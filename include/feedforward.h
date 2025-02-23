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

#include "cpu/feedforward.h"
