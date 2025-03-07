#include "metal/attention.h"

#include <cassert>


void Attention<BackendType::METAL>::forward(const RunParams& param) {
    int totlen = param.start_pos + param.seq_len;
    std::array<int, 3> grid_size = {param.batch, param.seq_len, param.dim};

    // wq(input), wk(input), wv(input)
    _executor->forward(obj_id, 3, param,
        {
            Attention_INPUT,
            Attention_WEIGHT_Q,
            Attention_WEIGHT_K,
            Attention_WEIGHT_V,
            Attention_XQ,
            Attention_CACHE_K,
            Attention_CACHE_V,
        },
        grid_size);

    // Rope apply_rotary_emb
    _executor->forward(obj_id, 4, param,
        {
            Rope_COST,
            Rope_SINT,
            Attention_XQ,
            Attention_CACHE_K,
        },
        {param.batch, param.seq_len, param.dim / 2});


    // softmax(QK^T/scale)) (not averaged)
    _executor->forward(obj_id, 5, param,
        {
            Attention_XQ,
            Attention_CACHE_K,
            Attention_CACHE_V,
            Attention_SCORE,
        },
        {param.batch, param.seq_len, totlen * param.num_heads});

    // score @ cachev
    _executor->forward(obj_id, 6, param,
        {
            Attention_SCORE,
            Attention_CACHE_V,
            Attention_OUTPUT,
        },
        grid_size);

    // Wo @ output
    if (!param.residual) {
        _executor->addBuffer(obj_id, Attention_RESIDUAL, obj_id, Attention_INPUT);  // dummy input  
    }
    _executor->forward(obj_id, 7, param,
        {
            Attention_OUTPUT,
            Attention_WEIGHT_O,
            Attention_RESIDUAL,
            Attention_RESULT,
        },
        grid_size);
}

void Attention<BackendType::METAL>::build(std::string_view content) {
    auto ptr = content.data();
    size_t weight_bytes = _dim * _dim * sizeof(float) + 12;
    _wq.build({ptr, static_cast<size_t>(weight_bytes)});
    ptr += weight_bytes;
    _wk.build({ptr, static_cast<size_t>(weight_bytes)});
    ptr += weight_bytes;
    _wv.build({ptr, static_cast<size_t>(weight_bytes)});
    ptr += weight_bytes;
    _wo.build({ptr, static_cast<size_t>(weight_bytes)});

    _cachek = Tensor({_executor->batch, _max_seq_len, _dim});
    _cachev = Tensor({_executor->batch, _max_seq_len, _dim});
    _executor->addBuffer(obj_id, Attention_CACHE_K, _cachek);
    _executor->addBuffer(obj_id, Attention_CACHE_V, _cachev);
    _executor->addBuffer(obj_id, Attention_WEIGHT_Q, _wq);
    _executor->addBuffer(obj_id, Attention_WEIGHT_K, _wk);
    _executor->addBuffer(obj_id, Attention_WEIGHT_V, _wv);
    _executor->addBuffer(obj_id, Attention_WEIGHT_O, _wo);
}

void Attention<BackendType::METAL>::alloc_shared_buffer(const RunParams& param) {
    // Assume seq_len in prefilling is more than seq_len in decoding.
    std::vector<int> input_shape = {param.batch, param.seq_len, param.dim};
    _executor->addBuffer(obj_id, Attention_XQ, input_shape);
    _executor->addBuffer(obj_id, Attention_SCORE, {param.batch, param.seq_len, param.max_seq_len, param.num_heads});
    _executor->addBuffer(obj_id, Attention_OUTPUT, input_shape);
    _executor->addBuffer(obj_id, Attention_RESULT, input_shape);
    _executor->addBuffer(obj_id, Rope_COST);
    _executor->addBuffer(obj_id, Rope_SINT);
}
