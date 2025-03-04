#pragma once

#include "include/attention.h"
#include "include/executor.h"
#include "include/backend.h"
#include "include/rope.h"
#include "include/feedforward.h"
#include "include/transformer.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>

template <>
class Transformer<BackendType::CPU> {

public:
    Transformer(const ModelArgs& args, std::shared_ptr<Executor<BackendType::CPU>> executor) {
        assert(args.dim % args.num_heads == 0);
        _args = args;

        for (int i = 0; i < args.num_layers; i++) {
            _attention.push_back(Attention<BackendType::CPU>(args.dim, args.num_heads, args.max_seq_len, executor));
            _ffn.push_back(FFNSwiGLU<BackendType::CPU>(args.dim, args.dim * 4, args.multiple_of));
        }

        _rope_cost = Tensor({args.max_seq_len, args.dim / args.num_heads / 2});
        _rope_sint = Tensor({args.max_seq_len, args.dim / args.num_heads / 2});
        precompute_freqs_cis(
            args.dim / args.num_heads, args.max_seq_len, 10000.0,
            &_rope_cost, &_rope_sint);
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

    Tensor forward(const std::vector<std::vector<int>>& input, int start_pos);

private:
    std::vector<Attention<BackendType::CPU>> _attention;
    std::vector<FFNSwiGLU<BackendType::CPU>> _ffn;
    Tensor _rope_cost;
    Tensor _rope_sint;
    Tensor _token_embeddings;
    Tensor _wo;

    ModelArgs _args;
};
