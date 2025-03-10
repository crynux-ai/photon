#include "include/profiler.h"
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
        grid_size
        PROFILE_TAG("Attention/WQKV"));

    // Rope apply_rotary_emb
    _executor->forward(obj_id, 4, param,
        {
            Rope_COST,
            Rope_SINT,
            Attention_XQ,
            Attention_CACHE_K,
        },
        {param.batch, param.seq_len, param.dim / 2}
        PROFILE_TAG("Attention/Rope_apply_emb"));


    // softmax(QK^T/scale)) (not averaged)
    _executor->forward(obj_id, 5, param,
        {
            Attention_XQ,
            Attention_CACHE_K,
            Attention_CACHE_V,
            Attention_SCORE,
        },
        {param.batch, param.seq_len, totlen * param.num_heads}
        PROFILE_TAG("Attention/Score"));

    // score @ cachev
    _executor->forward(obj_id, 6, param,
        {
            Attention_SCORE,
            Attention_CACHE_V,
            Attention_OUTPUT,
        },
        grid_size
        PROFILE_TAG("Attention/Score@V"));

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
        grid_size
        PROFILE_TAG("Attention/Output"));
}

void Attention<BackendType::METAL>::build(std::string_view content) {
    PROFILE_BEGIN(obj_id, "Attention/build")
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
    PROFILE_END(obj_id, "Attention/build")
}

void Attention<BackendType::METAL>::alloc_shared_buffer(const RunParams& param) {
    PROFILE_BEGIN(obj_id, "Attention/alloc")
    // Assume seq_len in prefilling is more than seq_len in decoding.
    size_t input_bytes = param.batch * param.seq_len * param.dim * sizeof(float);
    size_t score_bytes = param.batch * param.seq_len * param.max_seq_len * param.num_heads * sizeof(float);
    _executor->addBuffer(obj_id, Attention_XQ, input_bytes);
    _executor->addBuffer(obj_id, Attention_SCORE, score_bytes);
    _executor->addBuffer(obj_id, Attention_OUTPUT, input_bytes);
    _executor->addBuffer(obj_id, Attention_RESULT, input_bytes);
    _executor->useSharedBuffer(obj_id, Rope_COST);
    _executor->useSharedBuffer(obj_id, Rope_SINT);
    PROFILE_END(obj_id, "Attention/alloc")
}
