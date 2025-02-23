#pragma once

#include "include/backend.h"
#include "schema/tensor.h"
#include <cmath>

struct ModelArgs {
    int dim;
    int num_layers;
    int num_heads;
    int vocab_size;
    int multiple_of;
    float norm_eps = 1e-5;
    int max_seq_len = 2048;
};


template <BackendType backend>
class Transformer {

public:
    virtual ~Transformer() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual Tensor forward(const std::vector<std::vector<int>>& input, int start_pos) = 0;

protected:
    Transformer(const ModelArgs& args) = default;

};


template <> class Transformer<BackendType::CPU>;

template <> class Transformer<BackendType::METAL>;

#include "cpu/transformer.h"
