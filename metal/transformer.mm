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


void Transformer<BackendType::METAL>::forward(const RunParams& param) {
    Tensor embeddings({param.batch, param.seq_len, param.dim});
    Tensor result({param.batch, param.seq_len, param.vocab_size});
    size_t input_size = param.batch * param.seq_len * sizeof(float);
    size_t emb_size = param.batch * param.seq_len * param.dim * sizeof(float);
    size_t rope_size = param.max_seq_len * param.dim / 2 * sizeof(float);
    size_t result_size = param.batch * param.seq_len * param.vocab_size * sizeof(float);

    _executor->addBuffer(obj_id, Transformer_INPUT_EMBEDDING, embeddings._value.get(), emb_size);
    _executor->forward(obj_id, 8, param,
        {
            Transformer_INPUT,
            Transformer_EMBEDDING_TABLE,
            Transformer_INPUT_EMBEDDING,
        },
        {param.batch, param.seq_len, param.dim});
    _executor->bufferToTensor(obj_id, Transformer_INPUT_EMBEDDING, &embeddings);

    auto layer_param = param;
    for (int l = 0; l < _args.num_layers; l++) {
        layer_param.mask = param.seq_len > 1 ? 1 : 0;
        layer_param.residual = 1;
        // Attention
        auto norm_input = RMSNorm(embeddings, _args.norm_eps);
        _executor->addBuffer(_attention[l]->obj_id, Attention_INPUT, norm_input._value.get(), emb_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_RESIDUAL, embeddings._value.get(), emb_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_COST, _rope_cost._value.get(), rope_size);
        _executor->addBuffer(_attention[l]->obj_id, Attention_ROPE_SINT, _rope_sint._value.get(), rope_size);
        _attention[l]->forward(layer_param);
        _executor->bufferToTensor(_attention[l]->obj_id, Attention_RESULT, &embeddings);

        // FFN
        norm_input = RMSNorm(embeddings, _args.norm_eps);
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_INPUT, norm_input._value.get(), emb_size);
        _executor->addBuffer(_ffn[l]->obj_id, FFNSwiGLU_RESIDUAL, embeddings._value.get(), emb_size);
        _ffn[l]->forward(layer_param);
        _executor->bufferToTensor(_ffn[l]->obj_id, FFNSwiGLU_RESULT, &embeddings);
    }
    embeddings = RMSNorm(embeddings, _args.norm_eps);

    // Wo * emb
    _executor->addBuffer(obj_id, Transformer_OUTPUT, embeddings._value.get(), emb_size);
    _executor->addBuffer(obj_id, Transformer_RESULT, result._value.get(), result_size);
    _executor->forward(obj_id, 9, param,
        {
            Transformer_OUTPUT,
            Transformer_WEIGHT_O,
            Transformer_RESULT,
        },
        {param.batch, param.seq_len, param.vocab_size});
}

