#include "metal/transformer.h"

#include <cassert>
#include <cmath>


void Transformer<BackendType::METAL>::forward(const RunParams& param) {
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

        _attention[l]->alloc_shared_buffer(param);
        _executor->addBuffer(_attention[l]->obj_id, Attention_INPUT, obj_id, Transformer_INPUT_NORMS);
        _executor->addBuffer(_attention[l]->obj_id, Attention_RESIDUAL, obj_id, Transformer_INPUT_EMBEDDING);
        _attention[l]->forward(layer_param);
        _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, _attention[l]->obj_id, Attention_RESULT);

        // FFN
        _executor->forward(obj_id, 10, param,
            {
                Transformer_INPUT_EMBEDDING,
                Transformer_INPUT_NORMS,
            },
            {param.batch, param.seq_len, param.dim});
        _ffn[l]->alloc_shared_buffer(param);
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
    _executor->forward(obj_id, 9, param,
        {
            Transformer_OUTPUT,
            Transformer_WEIGHT_O,
            Transformer_RESULT,
        },
        {param.batch, param.seq_len, param.vocab_size});
}

void Transformer<BackendType::METAL>::build(std::string_view content) {
    auto ptr = content.data();
    auto attention_size = _attention[0]->size();
    auto ffn_size = _ffn[0]->size();

    for (int i = 0; i < _args.num_layers; i++) {
        _attention[i]->build({ptr, attention_size});
        ptr += attention_size;
        _ffn[i]->build({ptr, ffn_size});
        ptr += ffn_size;
    }

    size_t emb_size = _args.vocab_size * _args.dim * 4 + 12;
    _token_embeddings.build({ptr, emb_size});
    ptr += emb_size;
    _wo.build({ptr, emb_size});

    _executor->addBuffer(obj_id, Transformer_EMBEDDING_TABLE, _token_embeddings);
    _executor->addBuffer(obj_id, Transformer_WEIGHT_O, _wo);
}

void Transformer<BackendType::METAL>::alloc_shared_buffer(const RunParams& param) {
    std::vector<int> input_shape = {param.batch, param.seq_len, param.dim};
    _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, input_shape);
    _executor->addBuffer(obj_id, Transformer_INPUT_NORMS, input_shape);
    _executor->addBuffer(obj_id, Transformer_OUTPUT, obj_id, Transformer_INPUT_NORMS);
    _executor->addBuffer(obj_id, Transformer_RESULT, {param.batch, param.seq_len, param.vocab_size});
}

