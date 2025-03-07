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
        size_t weight_bytes = _dim * _dim * sizeof(float) + 12;
        size_t cache_bytes = _executor->batch * _max_seq_len * _dim * sizeof(float);

        _wq.build({ptr, static_cast<size_t>(weight_bytes)});
        ptr += weight_bytes;
        _wk.build({ptr, static_cast<size_t>(weight_bytes)});
        ptr += weight_bytes;
        _wv.build({ptr, static_cast<size_t>(weight_bytes)});
        ptr += weight_bytes;
        _wo.build({ptr, static_cast<size_t>(weight_bytes)});

        assert(_executor->batch > 0);
        _cachek = Tensor({_executor->batch, _max_seq_len, _dim});
        _cachev = Tensor({_executor->batch, _max_seq_len, _dim});

        
        weight_bytes -= 12;
        _executor->addBuffer(obj_id, Attention_CACHE_K, _cachek);
        _executor->addBuffer(obj_id, Attention_CACHE_V, _cachev);
        _executor->addBuffer(obj_id, Attention_WEIGHT_Q, _wq);
        _executor->addBuffer(obj_id, Attention_WEIGHT_K, _wk);
        _executor->addBuffer(obj_id, Attention_WEIGHT_V, _wv);
        _executor->addBuffer(obj_id, Attention_WEIGHT_O, _wo);
    }

    void forward(const RunParams& param);

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
