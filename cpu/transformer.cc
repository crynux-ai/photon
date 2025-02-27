#include "cpu/transformer.h"
#include "cpu/math_utils.h"
#include "cpu/norm.h"

#include <cassert>

Tensor Transformer<BackendType::CPU>::forward(const std::vector<std::vector<int>>& input, int start_pos) {
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
        embeddings = _attention[l].forward(norm_input, _rope_cost, _rope_sint, start_pos,
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
