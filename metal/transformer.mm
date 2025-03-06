#include "metal/transformer.h"

#include <cassert>
#include <cmath>

Tensor RMSNorm(const Tensor& x, float norm_eps) {
    assert (x.shape().size() == 3);
    auto batch = x.shape()[0];
    auto seqlen = x.shape()[1];
    auto dim = x.shape()[2];

    Tensor result(x.shape());
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seqlen; j++) {
            float sum = 0;
            for (int k = 0; k < dim; k++) {
                float tmp = x(i, j, k);
                sum += tmp * tmp;
            }
            sum /= dim;
            sum += norm_eps;
            sum = 1.0f / std::sqrt(sum);
            for (int k = 0; k < dim; k++) {
                result.set(x(i, j, k) * sum, i, j, k);
            }
        }
    }
    return result;
}


void Transformer<BackendType::METAL>::forward(int seqlen, int start_pos) {
    int batch = _executor->batch;
    Tensor embeddings({batch, seqlen, _args.dim});
    size_t input_size = batch * seqlen * sizeof(float);
    size_t emb_size = batch * seqlen * _args.dim * sizeof(float);
    size_t rope_size = _args.max_seq_len * _args.dim / 2 * sizeof(float);

    int params[] = {batch, seqlen, _args.max_seq_len, _args.dim, start_pos, _args.vocab_size};
    _executor->addBuffer(obj_id, Transformer_INPUT_PARAMS, params, 6 * sizeof(int));
    _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, embeddings._value.get(), emb_size);
    _executor->forward(obj_id, 8,
        {
            Transformer_INPUT_PARAMS,
            Transformer_INPUT,
            Transformer_EMBEDDING_TABLE,
            Transformer_INPUT_EMBEDDING,
        },
        {batch, seqlen, _args.dim});
    _executor->bufferToTensor(obj_id, Transformer_INPUT_EMBEDDING, &embeddings);

    for (int l = 0; l < _args.num_layers; l++) {
        // Attention
        auto norm_input = RMSNorm(embeddings, _args.norm_eps);
        _executor->addBuffer(_attention[l]->obj_id, Attention_INPUT, norm_input._value.get(), emb_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_RESIDUAL, embeddings._value.get(), emb_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_COST, _rope_cost._value.get(), rope_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_SINT, _rope_sint._value.get(), rope_size);
        _attention[l]->forward(seqlen, start_pos, /*mask=*/seqlen > 1, /*residual=*/true);
        _executor->bufferToTensor(_attention[l]->obj_id, Attention_RESULT, &embeddings);

        // FFN
        norm_input = RMSNorm(embeddings, _args.norm_eps);
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_INPUT, norm_input._value.get(), emb_size);
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_RESIDUAL, embeddings._value.get(), emb_size);
        _ffn[l]->forward(seqlen, true);
        _executor->bufferToTensor(_ffn[l]->obj_id, FFNSwiGLU_RESULT, &embeddings);
    }
    embeddings = RMSNorm(embeddings, _args.norm_eps);

    Tensor result({batch, seqlen, _args.vocab_size});
    size_t result_size = batch * seqlen * _args.vocab_size * sizeof(float);
    _executor->addBuffer(obj_id, Transformer_OUTPUT, embeddings._value.get(), emb_size);
    _executor->addBuffer(obj_id, Transformer_RESULT, result._value.get(), result_size);
    _executor->forward(obj_id, 9,
        {
            Transformer_INPUT_PARAMS,
            Transformer_OUTPUT,
            Transformer_WEIGHT_O,
            Transformer_RESULT,
        },
        {batch, seqlen, _args.vocab_size});
}

