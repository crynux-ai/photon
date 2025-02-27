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
    Transformer(const ModelArgs& args) {
        assert(args.dim % args.num_heads == 0);
        _args = args;

        for (int i = 0; i < args.num_layers; i++) {
            auto att = std::make_unique<Attention<BackendType::METAL>>(args.dim, args.num_heads);
            auto ffn = std::make_unique<FFNSwiGLU<BackendType::METAL>>(args.dim, args.dim * 4, args.multiple_of);

            _attention.push_back(std::move(att));
            _ffn.push_back(std::move(ffn));
        }
        _rope = precompute_freqs_cis(args.dim / args.num_heads, args.max_seq_len, 10000.0);
    }

    ~Transformer() {
        [_pipelineState release];
        [_kernelFunction release];
        [_library release];
        [_device release];
    }

    size_t size() {
        auto attention_size = _attention[0]->size();
        auto ffn_size = _ffn[0]->size();
        auto size = (attention_size + ffn_size) * _args.num_layers;
        size += (_args.vocab_size * _args.dim * 4 + 12) * 2;
        return size;
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto attention_size = _attention[0]->size();
        auto ffn_size = _ffn[0]->size();

        for (int i = 0; i < _args.num_layers; i++) {
            _attention[i]->build({ptr, attention_size});
            ptr += attention_size;
            _ffn[i]->build({ptr, ffn_size});
            ptr += ffn_size;
        }

        auto emb_size = _args.vocab_size * _args.dim * 4 + 12;
        _token_embeddings.build({ptr, static_cast<size_t>(emb_size)});
        ptr += emb_size;
        _wo.build({ptr, static_cast<size_t>(emb_size)});

        @autoreleasepool {
            NSError* error;
            _device = MTLCreateSystemDefaultDevice();
            if (!_device) {
                throw std::runtime_error("Metal is not supported");
            }
            NSString *libPath = [[NSBundle mainBundle] pathForResource:@"photon" ofType:@"metallib"];
            _library = [_device newLibraryWithFile:libPath error:&error];
            if (!_library) {
                throw std::runtime_error("Fail to load library");
            }

            _kernelFunction = [_library newFunctionWithName:@"FFNSwiGLU_step1"];
            if (!_kernelFunction) {
                throw std::runtime_error("Method not found in the library");
            }

            _pipelineState = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
            if (!_pipelineState) {
                throw std::runtime_error("Fail to create pipeline");
            }
        }
    }

    Tensor forward(const std::vector<std::vector<int>>& input, int start_pos);

private:
    std::vector<std::unique_ptr<Attention<BackendType::METAL>>> _attention;
    std::vector<std::unique_ptr<FFNSwiGLU<BackendType::METAL>>> _ffn;
    std::pair<FreqMatrix, FreqMatrix> _rope;
    Tensor _token_embeddings;
    Tensor _wo;

    ModelArgs _args;

    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    id<MTLFunction> _kernelFunction;
    id<MTLComputePipelineState> _pipelineState;
};
