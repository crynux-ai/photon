#include "metal/attention.h"

#include <cassert>


void Attention<BackendType::METAL>::forward(const RunParams& param) {
    int totlen = param.start_pos + param.seq_len;
    std::vector<int> input_shape = {param.batch, param.seq_len, param.dim};
    std::array<int, 3> grid_size = {param.batch, param.seq_len, param.dim};
    Tensor xq(input_shape);
    Tensor score({param.batch, param.seq_len, param.max_seq_len, param.num_heads});
    Tensor output(input_shape);
    Tensor result(input_shape);
    _executor->addBuffer(obj_id, Attention_XQ, xq);
    _executor->addBuffer(obj_id, Attention_SCORE, score);
    _executor->addBuffer(obj_id, Attention_OUTPUT, output);
    _executor->addBuffer(obj_id, Attention_RESULT, result);


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
            Attention_ROPE_COST,
            Attention_ROPE_SINT,
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
