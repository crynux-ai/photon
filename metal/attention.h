#pragma once

#include "include/backend.h"
#include "include/attention.h"
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
        [_state1 release];
        [_state2 release];
        [_state3 release];
        [_step1 release];
        [_step2 release];
        [_step3 release];
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

            _step1 = [_library newFunctionWithName:@"Attention_Step1"];
            _step2 = [_library newFunctionWithName:@"Rope_apply_rotary_emb"];
            _step3 = [_library newFunctionWithName:@"Attention_ComputeScore"];
            if (!_step1 || !_step2 || !_step3) {
                throw std::runtime_error("Method not found in the library");
            }

            _state1 = [_device newComputePipelineStateWithFunction:_step1 error:&error];
            _state2 = [_device newComputePipelineStateWithFunction:_step2 error:&error];
            _state3 = [_device newComputePipelineStateWithFunction:_step3 error:&error];
            if (!_state1 || !_state2 || !_state3) {
                throw std::runtime_error("Fail to create pipeline");
            }
        }
    }

    Tensor forward(
        const Tensor& input, const Tensor& cost, const Tensor& sint,
        int start_pos, bool mask, Tensor* residual=nullptr);

private:
    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    Tensor _cachek;
    Tensor _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;
    int _max_seq_len;

    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    id<MTLFunction> _step1;
    id<MTLFunction> _step2;
    id<MTLFunction> _step3;
    id<MTLComputePipelineState> _state1;
    id<MTLComputePipelineState> _state2;
    id<MTLComputePipelineState> _state3;

};
