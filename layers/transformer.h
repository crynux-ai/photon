#ifndef LAYERS_TRANSFORMER_H
#define LAYERS_TRANSFORMER_H

#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"
#include "layers/math_utils.h"
#include "layers/rope.h"
#include "layers/attention.h"
#include "layers/norm.h"
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
    Transformer(const ModelArgs& args) {
        assert(args.dim % args.num_heads == 0);
        _args = args;

        for (int i = 0; i < args.num_layers; i++) {
            _attention.push_back(Attention(args.dim, args.num_heads));
            _ffn.push_back(FFNSwiGLU<backend>(args.dim, args.dim * 4, args.multiple_of));
        }
        _rope = precompute_freqs_cis(args.dim / args.num_heads, args.max_seq_len, 10000.0);
    }

    size_t size() {
        auto attention_size = _attention[0].size();
        auto ffn_size = _ffn[0].size();
        auto size = (attention_size + ffn_size) * _args.num_layers;
        size += (_args.vocab_size * _args.dim * 4 + 12) * 2;
        return size;
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto attention_size = _attention[0].size();
        auto ffn_size = _ffn[0].size();

        for (int i = 0; i < _args.num_layers; i++) {
            _attention[i].build({ptr, attention_size});
            ptr += attention_size;
            _ffn[i].build({ptr, ffn_size});
            ptr += ffn_size;
        }

        auto emb_size = _args.vocab_size * _args.dim * 4 + 12;
        _token_embeddings.build({ptr, static_cast<size_t>(emb_size)});
        ptr += emb_size;
        _wo.build({ptr, static_cast<size_t>(emb_size)});
    }

    Tensor forward(const std::vector<std::vector<int>>& input, int start_pos) {
        int batch = input.size();
        int seqlen = input[0].size();

        Tensor embeddings({batch, seqlen, _args.dim});
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < seqlen; j++) {
                for (int k = 0; k  < _args.dim; k++) {
                    embeddings.set(_token_embeddings(input[i][j], k), i, j, k);
                }
            }
        }
        for (int l = 0; l < _args.num_layers; l++) {
            // Attention
            auto norm_input = RMSNorm(embeddings, _args.norm_eps);
            embeddings = _attention[l].forward(norm_input, _rope, start_pos,
                /*mask=*/seqlen > 1, /*residual=*/&embeddings);

            // FFN
            norm_input = RMSNorm(embeddings, _args.norm_eps);
            embeddings = _ffn[l].forward(norm_input, /*residual=*/&embeddings);
        }
        embeddings = RMSNorm(embeddings, _args.norm_eps);

        Tensor result({batch, seqlen, _args.vocab_size});
        result.zero();
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < seqlen; j++) {
                for (int v = 0; v < _args.vocab_size; v++) {
                    for (int k = 0; k < _args.dim; k++) {
                        result.add(embeddings(i, j, k) * _wo(v, k), i, j, v);
                    }
                }
            }
        }
        return result;
    }

private:
    std::vector<Attention> _attention;
    std::vector<FFNSwiGLU<backend>> _ffn;
    std::pair<FreqMatrix, FreqMatrix> _rope;
    Tensor _token_embeddings;
    Tensor _wo;

    ModelArgs _args;

};

#endif // LAYERS_TRANSFORMER_H