#include "metal/feedforward.h"

#include <cassert>


Tensor FFNSwiGLU<BackendType::METAL>::forward(const Tensor& input, Tensor* residual) {
    assert(input.shape().size() == 3);
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    assert(input.shape()[2] == _dim);

    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_pipelineState];
        
        size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
        size_t wByteSize = _dim * _hidden_dim * sizeof(float);
        size_t resByteSize = batch * seqlen * _hidden_dim * sizeof(float);
        int params[] = {batch, seqlen, _dim, _hidden_dim};
        Tensor r1({batch, seqlen, _hidden_dim});

        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:params length:4*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInput = [_device newBufferWithBytes:input._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferW1 = [_device newBufferWithBytes:_w1._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferW3 = [_device newBufferWithBytes:_w3._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferRes = [_device newBufferWithBytes:r1._value.get() length:resByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferW1 offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferW3 offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferRes offset:0 atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(batch, seqlen, _hidden_dim);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* tmp = static_cast<float*>(bufferRes.contents);
        memcpy(r1._value.get(), tmp, resByteSize);

        Tensor result(input.shape());
        result.zero();
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqlen; i++) {
                for (int j = 0; j < _dim; j++) {
                    if (residual) {
                        result.add((*residual)(b, i, j), b, i, j);
                    }
                    for (int k = 0; k < _hidden_dim; k++) {
                        result.add(r1(b, i, k) * _w2(j, k), b, i, j);
                    }
                }
            }
        }
        return result;
    }
    
}
