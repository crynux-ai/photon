#include "metal/attention.h"

#include <cassert>


Tensor Attention<BackendType::METAL>::forward(
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual) {
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    int num_complex = rope_cost.shape()[1];
    size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
    size_t wByteSize = _dim * _dim * sizeof(float);
    size_t qByteSize = batch * seqlen * _dim * sizeof(float);
    size_t cacheByteSize = batch * _max_seq_len * _dim * sizeof(float);
    if (_cachek.shape().empty()) {
        _cachek = Tensor({batch, _max_seq_len, _dim});
        _cachev = Tensor({batch, _max_seq_len, _dim});
        _bufferCachek = [_device newBufferWithBytes:_cachek._value.get() length:cacheByteSize options:MTLResourceStorageModeShared];
        _bufferCachev = [_device newBufferWithBytes:_cachev._value.get() length:cacheByteSize options:MTLResourceStorageModeShared];
    }

    @autoreleasepool {
        // wq(input), wk(input), wv(input)
        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state1];

        int params[] = {batch, seqlen, _max_seq_len, _dim, start_pos};
        Tensor xq({batch, seqlen, _dim});
        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:params length:5*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInput = [_device newBufferWithBytes:input._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWq = [_device newBufferWithBytes:_wq._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWk = [_device newBufferWithBytes:_wk._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWv = [_device newBufferWithBytes:_wv._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXq = [_device newBufferWithBytes:xq._value.get() length:qByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferWq offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferWk offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferWv offset:0 atIndex:4];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:5];
        [computeEncoder setBuffer:_bufferCachek offset:0 atIndex:6];
        [computeEncoder setBuffer:_bufferCachev offset:0 atIndex:7];
        
        MTLSize gridSize = MTLSizeMake(batch, seqlen, _dim);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Rope apply_rotary_emb
        commandQueue = [_device newCommandQueue];
        commandBuffer = [commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state2];

        size_t ropeByteSize = _max_seq_len * num_complex * sizeof(float);
        size_t xqByteSize = batch * seqlen * _dim * sizeof(float);
        int rope_params[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, num_complex};
        bufferParam = [_device newBufferWithBytes:rope_params length:7*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferCost = [_device newBufferWithBytes:rope_cost._value.get() length:ropeByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferSint = [_device newBufferWithBytes:rope_sint._value.get() length:ropeByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferCost offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferSint offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:3];
        [computeEncoder setBuffer:_bufferCachek offset:0 atIndex:4];

        gridSize = MTLSizeMake(batch, seqlen, _dim / 2);
        threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];


        // softmax(QK^T/scale)) (not averaged)
        int totlen = start_pos + seqlen;
        float scale = std::sqrt(_head_dim);
        int scoreByteSize = batch * seqlen * _max_seq_len * _num_heads * sizeof(float);
        Tensor score({batch, seqlen, _max_seq_len, _num_heads});
        Tensor output({batch, seqlen, _dim});

        commandQueue = [_device newCommandQueue];
        commandBuffer = [commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state3];

        int compute_score_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, mask ? 1 : 0};
        id<MTLBuffer> bufferParamScore = [_device newBufferWithBytes:compute_score_param length:7*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferScore = [_device newBufferWithBytes:score._value.get() length:scoreByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParamScore offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:1];
        [computeEncoder setBuffer:_bufferCachek offset:0 atIndex:2];
        [computeEncoder setBuffer:_bufferCachev offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferScore offset:0 atIndex:4];

        gridSize = MTLSizeMake(batch, seqlen, totlen * _num_heads);
        threadgroupSize = MTLSizeMake(1, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];


        // score @ cachev
        commandQueue = [_device newCommandQueue];
        commandBuffer = [commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state4];

        id<MTLBuffer> bufferOutput = [_device newBufferWithBytes:output._value.get() length:inputByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParamScore offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferScore offset:0 atIndex:1];
        [computeEncoder setBuffer:_bufferCachev offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferOutput offset:0 atIndex:3];


        gridSize = MTLSizeMake(batch, seqlen, _dim);
        threadgroupSize = MTLSizeMake(1, 1, 4);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Wo @ Output
        Tensor result({batch, seqlen, _dim});

        commandQueue = [_device newCommandQueue];
        commandBuffer = [commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state5];

        id<MTLBuffer> bufferResult = [_device newBufferWithBytes:result._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        int result_param[] = {batch, _max_seq_len, seqlen, start_pos, _dim, _num_heads, residual ? 1 : 0};
        id<MTLBuffer> bufferParamResult = [_device newBufferWithBytes:result_param length:7*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWo = [_device newBufferWithBytes:_wo._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResidual;
        if (residual) {
            bufferResidual = [_device newBufferWithBytes:residual->_value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        } else {
            bufferResidual = bufferInput; // dummy
        }

        [computeEncoder setBuffer:bufferParamResult offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferOutput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferWo offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferResidual offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferResult offset:0 atIndex:4];

        gridSize = MTLSizeMake(batch, seqlen, _dim);
        threadgroupSize = MTLSizeMake(1, 1, 4);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* ptr = static_cast<float*>(bufferResult.contents);
        memcpy(result._value.get(), ptr, batch * seqlen * _dim * sizeof(float));
        return result;
    }
}
