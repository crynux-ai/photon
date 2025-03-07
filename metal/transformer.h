#pragma once

#include "include/backend.h"
#include "include/attention.h"
#include "include/feedforward.h"
#include "include/transformer.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


template <>
class Transformer<BackendType::METAL> {

public:
    int obj_id;

    Transformer(const ModelArgs& args, std::shared_ptr<Executor<BackendType::METAL>> executor) {
        assert(args.dim % args.num_heads == 0);
        _args = args;
        _executor = executor;
        obj_id = executor->nextLayerId();

        for (int i = 0; i < args.num_layers; i++) {
            auto att = std::make_unique<Attention<BackendType::METAL>>(
                args.dim, args.num_heads, args.max_seq_len, executor);
            auto ffn = std::make_unique<FFNSwiGLU<BackendType::METAL>>(
                args.dim, args.dim * 4, args.multiple_of, executor);

            _attention.push_back(std::move(att));
            _ffn.push_back(std::move(ffn));
        }
        _rope_cost = Tensor({args.max_seq_len, args.dim / args.num_heads / 2});
        _rope_sint = Tensor({args.max_seq_len, args.dim / args.num_heads / 2});
        precompute_freqs_cis(
            args.dim / args.num_heads, args.max_seq_len, 10000.0,
            &_rope_cost, &_rope_sint);
    }

    ~Transformer() {}

    size_t size() {
        auto attention_size = _attention[0]->size();
        auto ffn_size = _ffn[0]->size();
        auto size = (attention_size + ffn_size) * _args.num_layers;
        size += (_args.vocab_size * _args.dim * 4 + 12) * 2;
        return size;
    }

    void build(std::string_view content);

    void forward(const RunParams& param);

    void alloc_shared_buffer(const RunParams& param);

private:
    std::shared_ptr<Executor<BackendType::METAL>> _executor;
    std::vector<std::unique_ptr<Attention<BackendType::METAL>>> _attention;
    std::vector<std::unique_ptr<FFNSwiGLU<BackendType::METAL>>> _ffn;
    Tensor _rope_cost;
    Tensor _rope_sint;
    Tensor _token_embeddings;
    Tensor _wo;

    ModelArgs _args;
};
