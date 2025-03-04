#pragma once

#include "include/backend.h"
#include "include/executor.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <map>


template <>
class Executor<BackendType::METAL> {

public:
    Executor(int batch) : batch(batch) {
        _func_names = {
            {1, "FFNSwiGLU_Step1"},
            {2, "FFNSwiGLU_Step2"},
            {3, "Attention_Step1"},
            {4, "Rope_apply_rotary_emb"},
            {5, "Attention_ComputeScore"},
            {6, "Attention_Output"},
            {7, "Attention_Result"},
            {8, "Transformer_TokenEmbedding"},
            {9, "Transformer_Result"},
        };
        _next_layer_id = 0;
    }

    int batch;

    ~Executor() {
        for (const auto& pair : _states) {
            [pair.second release];
        }
        for (const auto& pair : _funcs) {
            [pair.second release];
        }
        for (const auto& obj: _buffer) {
            for (const auto& pair: obj.second) {
                [pair.second release];
            }
        }

        [_library release];
        [_device release];
    }

    void build() {
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

            for (const auto& pair : _func_names) {
                NSString* func_name = [[NSString alloc] initWithUTF8String:pair.second.c_str()];
                _funcs[pair.first] = [_library newFunctionWithName: func_name];
                if (!_funcs[pair.first]) {
                    throw std::runtime_error("Method not found in the library");    
                }
                _states[pair.first] = [_device newComputePipelineStateWithFunction:_funcs[pair.first] error:&error];
                if (!_states[pair.first]) {
                    throw std::runtime_error("Fail to create pipeline");
                }
            }
        }
    }

    void forward(int obj_id, int func, std::vector<int> command_args, std::array<int, 3> grid_size) {
        @autoreleasepool {
            _command_queue = [_device newCommandQueue];
            _command_buffer = [_command_queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [_command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:_states[func]];

            for (int i = 0; i < command_args.size(); i++) {
                [encoder setBuffer:_buffer[obj_id][command_args[i]] offset:0 atIndex:i];
            }

            MTLSize gridSize = MTLSizeMake(grid_size[0], grid_size[1], grid_size[2]);
            MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [_command_buffer commit];
            [_command_buffer waitUntilCompleted];
        }
    }

    void addBuffer(int obj_id, int idx, void* data_ptr, size_t data_size) {
        auto tmp = [_device newBufferWithBytes:data_ptr length:data_size options:MTLResourceStorageModeShared];
        if (_buffer.find(obj_id) == _buffer.end()) {
            _buffer.insert({obj_id, {{idx, tmp}}});
        } else {
            _buffer[obj_id][idx] = tmp;
        }
    }

    void bufferToTensor(int obj_id, int idx, Tensor* tensor) {
        float* ptr = static_cast<float*>(_buffer[obj_id][idx].contents);
        size_t cnt = 1;
        for (int d : tensor->shape()) {
            cnt *= d;
        }
        memcpy(tensor->_value.get(), ptr, cnt * sizeof(float));
    }

    int nextLayerId() {
        return _next_layer_id++;
    }

    int _next_layer_id;

    id<MTLDevice> _device;
    id<MTLLibrary> _library;
    id<MTLCommandQueue> _command_queue;
    id<MTLCommandBuffer> _command_buffer;

    std::map<int, std::map<int, id<MTLBuffer>>> _buffer;

    std::map<int, std::string> _func_names;
    std::map<int, id<MTLFunction>> _funcs;
    std::map<int, id<MTLComputePipelineState>> _states;
};

