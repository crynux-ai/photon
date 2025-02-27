#include "metal/feedforward.h"

#include <cassert>


Tensor FFNSwiGLU<BackendType::METAL>::forward(const Tensor& input, Tensor* residual) {
    assert(input.shape().size() == 3);
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    assert(input.shape()[2] == _dim);

    @autoreleasepool {
        size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
        size_t wByteSize = _dim * _hidden_dim * sizeof(float);
        size_t resByteSize = batch * seqlen * _hidden_dim * sizeof(float);
        int params[] = {batch, seqlen, _dim, _hidden_dim};
        Tensor interres({batch, seqlen, _hidden_dim});

        // silu(w1(x)) * w3(x)
        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state1];

        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:params length:4*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInput = [_device newBufferWithBytes:input._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferW1 = [_device newBufferWithBytes:_w1._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferW3 = [_device newBufferWithBytes:_w3._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInterres = [_device newBufferWithBytes:interres._value.get() length:resByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferW1 offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferW3 offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferInterres offset:0 atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(batch, seqlen, _hidden_dim);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // w2(res)
        commandQueue = [_device newCommandQueue];
        commandBuffer = [commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state2];

        Tensor output(input.shape());
        int params2[] = {batch, seqlen, _dim, _hidden_dim, residual ? 1 : 0};

        bufferParam = [_device newBufferWithBytes:params2 length:5*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferW2 = [_device newBufferWithBytes:_w2._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferOutput = [_device newBufferWithBytes:output._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResidual;
        if (residual) {
            bufferResidual = [_device newBufferWithBytes:residual->_value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        } else {
            bufferResidual = bufferInput;  // Dummy input.
        }

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferResidual offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferW2 offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferInterres offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferOutput offset:0 atIndex:4];
        
        gridSize = MTLSizeMake(batch, seqlen, _dim);
        threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* tmp = static_cast<float*>(bufferOutput.contents);
        memcpy(output._value.get(), tmp, inputByteSize);
        return output;
    }
    
}
