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

    void build(std::string_view content);

    void forward(const RunParams& param);

    void alloc_shared_buffer(const RunParams& param);

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
