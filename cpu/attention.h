#pragma once

#include "include/backend.h"
#include "include/executor.h"
#include "include/attention.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>


template <>
class Attention<BackendType::CPU> {

public:
    Attention(int dim, int num_heads, int maxlen, std::shared_ptr<Executor<BackendType::CPU>> executor) {
        _dim = dim;
        _num_heads = num_heads;
        _head_dim = dim / num_heads;
        _maxlen = maxlen;
        _executor = executor;
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
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual=nullptr);

private:
    std::shared_ptr<Executor<BackendType::CPU>> _executor;

    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    Tensor _cachek;
    Tensor _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;
    int _maxlen;
};
