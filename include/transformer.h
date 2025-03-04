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

enum TransformerTensor {
    Transformer_INPUT = 0,
    Transformer_EMBEDDING_TABLE = 1,
    Transformer_WEIGHT_O = 2,
    Transformer_INPUT_EMBEDDING = 3,
    Transformer_INPUT_PARAMS = 4,
    Transformer_OUTPUT = 5,
    Transformer_RESULT = 6,
};



template <BackendType backend>
class Transformer {

public:
    virtual ~Transformer() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual Tensor forward(const Tensor& input, int start_pos) = 0;

protected:
    Transformer(const ModelArgs& args) = default;

};


template <> class Transformer<BackendType::CPU>;

template <> class Transformer<BackendType::METAL>;

#ifdef PHOTON_METAL
#include "metal/transformer.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/transformer.h"
    #endif
#endif

