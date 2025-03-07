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

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_read_size = _dim * _actual_hidden_dim * 4 + 12;

        _w1.build({ptr, static_cast<size_t>(weight_read_size)});
        ptr += weight_read_size;
        _w2.build({ptr, static_cast<size_t>(weight_read_size)});
        ptr += weight_read_size;
        _w3.build({ptr, static_cast<size_t>(weight_read_size)});

        size_t weight_size = _dim * _actual_hidden_dim * sizeof(float);

        _executor->addBuffer(obj_id, FFNSwiGLU_W1, _w1);
        _executor->addBuffer(obj_id, FFNSwiGLU_W2, _w2);
        _executor->addBuffer(obj_id, FFNSwiGLU_W3, _w3);
    }

    void forward(const RunParams& param);

private:
    std::shared_ptr<Executor<BackendType::METAL>> _executor;

    Tensor _w1;
    Tensor _w2;
    Tensor _w3;
    int _dim;
    int _actual_hidden_dim;
};

