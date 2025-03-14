#include "include/rope.h"
#include "include/params.h"

#include <cassert>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

template<>
void apply_rotary_emb<BackendType::METAL>(
        Tensor* xq,
        Tensor* cachek,
        const Tensor& cost,
        const Tensor& sint,
        int start_pos,
        int seqlen) {
    int batch = xq->shape()[0];
    int dim = xq->shape()[2];
    int max_seq_len = cost.shape()[0];
    int num_complex = cost.shape()[1];
    int head_dim = num_complex * 2;
    int num_head = dim / head_dim;
    assert(num_head * head_dim == dim);

     @autoreleasepool {
        NSError* error;
        id<MTLDevice> _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            throw std::runtime_error("Metal is not supported");
        }
        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"photon" ofType:@"metallib"];
        id<MTLLibrary> _library = [_device newLibraryWithFile:libPath error:&error];
        if (!_library) {
            throw std::runtime_error("Fail to load library");
        }

        id<MTLFunction> _kernelFunction = [_library newFunctionWithName:@"Rope_apply_rotary_emb"];
        if (!_kernelFunction) {
            throw std::runtime_error("Method not found in the library");
        }

        id<MTLComputePipelineState> _pipelineState = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        if (!_pipelineState) {
            throw std::runtime_error("Fail to create pipeline");
        }

        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_pipelineState];

        RunParams param = {
            .batch = batch,
            .seq_len = seqlen,
            .max_seq_len = max_seq_len,
            .start_pos = start_pos,
            .dim = dim,
            .num_heads = num_head,
            .head_dim = head_dim,
            .num_complex = num_complex,
        };

        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:&param length:sizeof(param) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferCost = [_device newBufferWithBytes:cost.data() length:cost.bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferSint = [_device newBufferWithBytes:sint.data() length:sint.bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXq = [_device newBufferWithBytes:xq->data() length:xq->bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferCachek = [_device newBufferWithBytes:cachek->data() length:cachek->bytes() options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferCost offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferSint offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferCachek offset:0 atIndex:4];

        MTLSize gridSize = MTLSizeMake(batch, seqlen, dim / 2);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* ptrq = static_cast<float*>(bufferXq.contents);
        memcpy(xq->data(), ptrq, xq->bytes());

        float* ptrk = static_cast<float*>(bufferCachek.contents);
        int batch_cnt = max_seq_len * dim;
        int copy_byte_size = seqlen * dim * sizeof(float);
        int ptr = start_pos * dim;
        for (int b = 0; b < batch; b++, ptr += batch_cnt) {
            memcpy(cachek->data() + ptr, ptrk + ptr, copy_byte_size);
        }
    }
}
