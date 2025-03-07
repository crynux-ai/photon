#pragma once

#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


template <>
class FFNSwiGLU<BackendType::METAL> {

public:
    int obj_id;

    inline static int calc_hidden_dim(int hidden_dim, int multiple_of) {
        return multiple_of * ((2 * hidden_dim / 3 + multiple_of - 1) / multiple_of);
    }

    FFNSwiGLU(int dim, int hidden_dim, int multiple_of,
              std::shared_ptr<Executor<BackendType::METAL>> executor) {
        _dim = dim;
        _actual_hidden_dim = calc_hidden_dim(hidden_dim, multiple_of);
        _executor = executor;
        obj_id = executor->nextLayerId();
    }

    ~FFNSwiGLU() {}

    size_t size() {
        return 3 * (_dim * _actual_hidden_dim * 4 + 12);
    }

    void build(std::string_view content);

    void forward(const RunParams& param);

    void alloc_shared_buffer(const RunParams& param);

private:
    std::shared_ptr<Executor<BackendType::METAL>> _executor;

    Tensor _w1;
    Tensor _w2;
    Tensor _w3;
    int _dim;
    int _actual_hidden_dim;
};

