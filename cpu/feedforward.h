#pragma once

#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>

template <>
class FFNSwiGLU<BackendType::CPU> {

public:
    FFNSwiGLU(int dim, int hidden_dim, int multiple_of, std::shared_ptr<Executor<BackendType::CPU>> executor) {
        _dim = dim;
        _hidden_dim = multiple_of * ((2 * hidden_dim / 3 + multiple_of - 1) / multiple_of);
        _executor = executor;
    }

    size_t size() {
        return 3 * (_dim * _hidden_dim * 4 + 12);
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_size = _dim * _hidden_dim * 4 + 12;

        _w1.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _w2.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _w3.build({ptr, static_cast<size_t>(weight_size)});
    }

    Tensor forward(const Tensor& input, Tensor* residual=nullptr);

private:
    std::shared_ptr<Executor<BackendType::CPU>> _executor;
    Tensor _w1;
    Tensor _w2;
    Tensor _w3;
    int _dim;
    int _hidden_dim;
};
