#include "metal/attention.h"

#include <cassert>


Tensor Attention<BackendType::METAL>::forward(
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual) {
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    int num_complex = rope_cost.shape()[1];
    if (_cachek.shape().empty()) {
        _cachek = Tensor({batch, _max_seq_len, _dim});
        _cachev = Tensor({batch, _max_seq_len, _dim});
    }

    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_state1];
        
        // wq(input), wk(input), wv(input)
        size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
        size_t wByteSize = _dim * _dim * sizeof(float);
        size_t qByteSize = batch * seqlen * _dim * sizeof(float);
        size_t cacheByteSize = batch * _max_seq_len * _dim * sizeof(float);
        int params[] = {batch, seqlen, _max_seq_len, _dim, start_pos};
        Tensor xq({batch, seqlen, _dim});

        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:params length:5*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInput = [_device newBufferWithBytes:input._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWq = [_device newBufferWithBytes:_wq._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWk = [_device newBufferWithBytes:_wk._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWv = [_device newBufferWithBytes:_wv._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXq = [_device newBufferWithBytes:xq._value.get() length:qByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferCachek = [_device newBufferWithBytes:_cachek._value.get() length:cacheByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferCachev = [_device newBufferWithBytes:_cachev._value.get() length:cacheByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferWq offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferWk offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferWv offset:0 atIndex:4];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:5];
        [computeEncoder setBuffer:bufferCachek offset:0 atIndex:6];
        [computeEncoder setBuffer:bufferCachev offset:0 atIndex:7];
        
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
        [computeEncoder setBuffer:bufferCachek offset:0 atIndex:4];

        gridSize = MTLSizeMake(batch, seqlen, _dim / 2);
        threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        int batch_cnt = _max_seq_len * _dim;
        int copy_byte_size = seqlen * _dim * sizeof(float);
        int ptr = start_pos * _dim;


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
        [computeEncoder setBuffer:bufferCachek offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferCachev offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferScore offset:0 atIndex:4];

        gridSize = MTLSizeMake(batch, seqlen, totlen * _num_heads);
        threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* ptrs = static_cast<float*>(bufferScore.contents);
        memcpy(score._value.get(), ptrs, batch * seqlen * _max_seq_len * _num_heads * sizeof(float));


        // score @ cachev
        float* ptrq = static_cast<float*>(bufferXq.contents);
        memcpy(xq._value.get(), ptrq, qByteSize);
        float* ptrk = static_cast<float*>(bufferCachek.contents);
        float* ptrv = static_cast<float*>(bufferCachev.contents);
        for (int b = 0; b < batch; b++, ptr += batch_cnt) {
            memcpy(_cachek._value.get() + ptr, ptrk + ptr, copy_byte_size);
            memcpy(_cachev._value.get() + ptr, ptrv + ptr, copy_byte_size);
        }

        output.zero();
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqlen; i++) {
                for (int j = 0; j < _dim; j++) {
                    float sum = 0;
                    for (int k = 0; k < totlen; k++) {
                        sum += score(b, i, k, j / _head_dim);
                    }
                    for (int k = 0; k < totlen; k++) {
                        output.add(score(b, i, k, j / _head_dim) / sum * _cachev(b, k, j), b, i, j);
                    }
                }
            }
        }

        Tensor result({batch, seqlen, _dim});
        result.zero();
        for (int b = 0; b < batch; b++) {
            for (int l = 0;  l < seqlen; l++) {
                for (int i = 0; i < _dim; i++) {
                    for (int j = 0; j < _dim; j++) {
                        result.add(_wo(i, j) * output(b, l, j), b, l, i);
                    }
                    if (residual) {
                        result.add((*residual)(b, l, i), b, l, i);
                    }
                }
            }
        }
        return result;
    }
}
