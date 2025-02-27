#pragma once

#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


template <>
class Attention<BackendType::METAL> {

public:
    Attention(int dim, int num_heads, int max_seq_len) {
        _dim = dim;
        _num_heads = num_heads;
        _head_dim = dim / num_heads;
        _max_seq_len = max_seq_len;
        assert(dim % num_heads == 0);
    }

    ~Attention() {
        [_pipelineState release];
        [_kernelFunction release];
        [_library release];
        [_device release];
    }

    size_t size() {
        return 4 * (_dim * _dim * 4 + 12);
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_size = _dim * _dim * 4 + 12;

        _wq.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wk.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wv.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _wo.build({ptr, static_cast<size_t>(weight_size)});

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

            _kernelFunction = [_library newFunctionWithName:@"Attention_Step1"];
            if (!_kernelFunction) {
                throw std::runtime_error("Method not found in the library");
            }

            _pipelineState = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
            if (!_pipelineState) {
                throw std::runtime_error("Fail to create pipeline");
            }
        }
    }

    Tensor forward(
        const Tensor& input, const std::pair<FreqMatrix, FreqMatrix>& rope,
        int start_pos, bool mask, Tensor* residual=nullptr);

private:
    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    std::vector<std::vector<Tensor>> _cachek;
    std::vector<std::vector<Tensor>> _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;
    int _max_seq_len;

    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    id<MTLFunction> _kernelFunction;
    id<MTLComputePipelineState> _pipelineState;
};
