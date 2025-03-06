#include "metal/transformer.h"

#include <cassert>
#include <cmath>


void Transformer<BackendType::METAL>::forward(const RunParams& param) {
    Tensor embeddings({param.batch, param.seq_len, param.dim});
    Tensor norms({param.batch, param.seq_len, param.dim});
    Tensor result({param.batch, param.seq_len, param.vocab_size});
    size_t input_size = param.batch * param.seq_len * sizeof(float);
    size_t emb_size = param.batch * param.seq_len * param.dim * sizeof(float);
    size_t rope_size = param.max_seq_len * param.num_complex * sizeof(float);
    size_t result_size = param.batch * param.seq_len * param.vocab_size * sizeof(float);

    _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, embeddings._value.get(), emb_size);
    _executor->addBuffer(obj_id, Transformer_INPUT_NORMS, norms._value.get(), emb_size);

    _executor->forward(obj_id, 8, param,
        {
            Transformer_INPUT,
            Transformer_EMBEDDING_TABLE,
            Transformer_INPUT_EMBEDDING,
        },
        {param.batch, param.seq_len, param.dim});

    auto layer_param = param;
    for (int l = 0; l < _args.num_layers; l++) {
        layer_param.mask = param.seq_len > 1 ? 1 : 0;
        layer_param.residual = 1;
        // Attention
        _executor->forward(obj_id, 10, param,
            {
                Transformer_INPUT_EMBEDDING,
                Transformer_INPUT_NORMS,
            },
            {param.batch, param.seq_len, param.dim});

        _executor->addBuffer(_attention[l]->obj_id, Attention_INPUT, obj_id, Transformer_INPUT_NORMS);
        _executor->addBuffer(_attention[l]->obj_id, Attention_RESIDUAL, obj_id, Transformer_INPUT_EMBEDDING);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_COST, _rope_cost._value.get(), rope_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_SINT, _rope_sint._value.get(), rope_size);
        _attention[l]->forward(layer_param);
        _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, _attention[l]->obj_id, Attention_RESULT);

        // FFN
        _executor->forward(obj_id, 10, param,
            {
                Transformer_INPUT_EMBEDDING,
                Transformer_INPUT_NORMS,
            },
            {param.batch, param.seq_len, param.dim});
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_INPUT, obj_id, Transformer_INPUT_NORMS);
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_RESIDUAL, obj_id, Transformer_INPUT_EMBEDDING);
        _ffn[l]->forward(layer_param);
        _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, _ffn[l]->obj_id, FFNSwiGLU_RESULT);
    }
    _executor->forward(obj_id, 10, param,
            {
                Transformer_INPUT_EMBEDDING,
                Transformer_INPUT_NORMS,
            },
            {param.batch, param.seq_len, param.dim});

    // Wo * emb
    _executor->addBuffer(obj_id, Transformer_OUTPUT, obj_id, Transformer_INPUT_NORMS);
    _executor->addBuffer(obj_id, Transformer_RESULT, result._value.get(), result_size);
    _executor->forward(obj_id, 9, param,
        {
            Transformer_OUTPUT,
            Transformer_WEIGHT_O,
            Transformer_RESULT,
        },
        {param.batch, param.seq_len, param.vocab_size});
}

