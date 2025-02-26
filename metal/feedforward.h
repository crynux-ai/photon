#pragma once

#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


template <>
class FFNSwiGLU<BackendType::METAL> {

public:
    FFNSwiGLU(int dim, int hidden_dim, int multiple_of) {
        _dim = dim;
        _hidden_dim = multiple_of * ((2 * hidden_dim / 3 + multiple_of - 1) / multiple_of);
    }

    ~FFNSwiGLU() {
        [_pipelineState release];
        [_kernelFunction release];
        [_library release];
        [_device release];
    }

    size_t size() {
        return 3 * (_dim * _hidden_dim * 4 + 12);
    }

    void build(std::string_view content) {
        auto ptr = content.data();
        auto weight_size = _dim * _hidden_dim * 4 + 12;

        _w1.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _w2.build({ptr, static_cast<size_t>(weight_size)});
        ptr += weight_size;
        _w3.build({ptr, static_cast<size_t>(weight_size)});

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

    Tensor forward(const Tensor& input, Tensor* residual=nullptr);

private:
    Tensor _w1;
    Tensor _w2;
    Tensor _w3;
    int _dim;
    int _hidden_dim;

    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    id<MTLFunction> _kernelFunction;
    id<MTLComputePipelineState> _pipelineState;
};

