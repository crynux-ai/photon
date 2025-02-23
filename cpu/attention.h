#pragma once

#include "include/backend.h"
#include "include/attention.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>


template <>
class Attention<BackendType::CPU> {

public:
    Attention(int dim, int num_heads) {
        _dim = dim;
        _num_heads = num_heads;
        _head_dim = dim / num_heads;
        assert(dim % num_heads == 0);
    }

    size_t size() {
        return 4 * (_dim * _dim * 4 + 12);
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_size = _dim * _dim * 4 + 12;

        _wq.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wk.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wv.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wo.build({ptr, static_cast<size_t>(weight_size)});
    }

    Tensor forward(
        const Tensor& input, const std::pair<FreqMatrix, FreqMatrix>& rope,
        int start_pos, bool mask, Tensor* residual=nullptr);

private:
    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    std::vector<std::vector<Tensor>> _cachek;
    std::vector<std::vector<Tensor>> _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;
};
