#pragma once

#include "include/backend.h"
#include "include/attention.h"
#include "include/executor.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


template <>
class Attention<BackendType::METAL> {

public:
    int obj_id;

    Attention(int dim, int num_heads, int max_seq_len,
              std::shared_ptr<Executor<BackendType::METAL>> executor) {
        _dim = dim;
        _num_heads = num_heads;
        _head_dim = dim / num_heads;
        _max_seq_len = max_seq_len;
        _executor = executor;
        obj_id = executor->nextLayerId();
        assert(dim % num_heads == 0);
    }

    ~Attention() {}

    size_t size() {
        return 4 * (_dim * _dim * 4 + 12);
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_read_size = _dim * _dim * 4 + 12;

        _wq.build({ptr, static_cast<size_t>(weight_read_size)});
        ptr += weight_read_size;
        _wk.build({ptr, static_cast<size_t>(weight_read_size)});
        ptr += weight_read_size;
        _wv.build({ptr, static_cast<size_t>(weight_read_size)});
        ptr += weight_read_size;
        _wo.build({ptr, static_cast<size_t>(weight_read_size)});

        assert(_executor->batch > 0);
        _cachek = Tensor({_executor->batch, _max_seq_len, _dim});
        _cachev = Tensor({_executor->batch, _max_seq_len, _dim});

        size_t cache_size = _executor->batch * _max_seq_len * _dim * sizeof(float);
        size_t weight_size = _dim * _dim * sizeof(float);

        _executor->addBuffer(obj_id, Attention_CACHE_K, _cachek._value.get(), cache_size);
        _executor->addBuffer(obj_id, Attention_CACHE_V, _cachev._value.get(), cache_size);
        _executor->addBuffer(obj_id, Attention_WEIGHT_Q, _wq._value.get(), weight_size);
        _executor->addBuffer(obj_id, Attention_WEIGHT_K, _wk._value.get(), weight_size);
        _executor->addBuffer(obj_id, Attention_WEIGHT_V, _wv._value.get(), weight_size);
        _executor->addBuffer(obj_id, Attention_WEIGHT_O, _wo._value.get(), weight_size);
    }

    Tensor forward(
        const Tensor& input, const Tensor& cost, const Tensor& sint,
        int start_pos, bool mask, Tensor* residual=nullptr);

private:
    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    Tensor _cachek;
    Tensor _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;
    int _max_seq_len;

    std::shared_ptr<Executor<BackendType::METAL>> _executor;

    id<MTLBuffer> _bufferCachev;
    id<MTLBuffer> _bufferCachek;

};
