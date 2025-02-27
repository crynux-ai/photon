#include "cpu/attention.h"

#include <cassert>


Tensor Attention<BackendType::METAL>::forward(
        const Tensor& input, const std::pair<FreqMatrix, FreqMatrix>& rope,
        int start_pos, bool mask, Tensor* residual) {
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    if (_cachek.empty()) {
        _cachek.resize(batch);
        _cachev.resize(batch);
    }

    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = [_device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_pipelineState];
        
        size_t inputByteSize = batch * seqlen * _dim * sizeof(float);
        size_t wByteSize = _dim * _dim * sizeof(float);
        size_t resByteSize = batch * seqlen * _dim * sizeof(float);
        int params[] = {batch, seqlen, _dim};
        Tensor xq({batch, seqlen, _dim});
        Tensor xk({batch, seqlen, _dim});
        Tensor xv({batch, seqlen, _dim});

        id<MTLBuffer> bufferParam = [_device newBufferWithBytes:params length:3*sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferInput = [_device newBufferWithBytes:input._value.get() length:inputByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWq = [_device newBufferWithBytes:_wq._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWk = [_device newBufferWithBytes:_wk._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferWv = [_device newBufferWithBytes:_wv._value.get() length:wByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXq = [_device newBufferWithBytes:xq._value.get() length:resByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXk = [_device newBufferWithBytes:xk._value.get() length:resByteSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferXv = [_device newBufferWithBytes:xv._value.get() length:resByteSize options:MTLResourceStorageModeShared];

        [computeEncoder setBuffer:bufferParam offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferInput offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferWq offset:0 atIndex:2];
        [computeEncoder setBuffer:bufferWk offset:0 atIndex:3];
        [computeEncoder setBuffer:bufferWv offset:0 atIndex:4];
        [computeEncoder setBuffer:bufferXq offset:0 atIndex:5];
        [computeEncoder setBuffer:bufferXk offset:0 atIndex:6];
        [computeEncoder setBuffer:bufferXv offset:0 atIndex:7];
        
        MTLSize gridSize = MTLSizeMake(batch, seqlen, _dim);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 16);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* ptrq = static_cast<float*>(bufferXq.contents);
        float* ptrk = static_cast<float*>(bufferXk.contents);
        float* ptrv = static_cast<float*>(bufferXv.contents);
        std::vector<std::vector<Tensor>> cacheq(batch);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqlen; i++) {
                _cachek[b].push_back(Tensor({_dim}));
                _cachev[b].push_back(Tensor({_dim}));
                cacheq[b].push_back(Tensor({_dim}));
                for (int j = 0; j < _dim; j++, ptrq++, ptrk++, ptrv++) {
                    cacheq[b].back().set(*ptrq, j);
                    _cachek[b].back().set(*ptrk, j);
                    _cachev[b].back().set(*ptrv, j);
                }
            }
        }
        
        apply_rotary_emb(&cacheq, &_cachek, rope.first, rope.second, start_pos, seqlen);

        int totlen = start_pos + seqlen;
        float scale = std::sqrt(_head_dim);
        std::vector<std::vector<Tensor>> score(batch);
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < _num_heads; h++) {
                score[b].push_back(Tensor({seqlen, totlen}));
                score[b].back().zero();
                int fill = totlen;
                if (mask) {
                    fill = start_pos;
                }
                for (int i = 0;  i < seqlen; i++) {
                    if (mask) {
                        fill++;
                        if (fill > totlen) {
                            fill = totlen;
                        }
                    }
                    float sum = 0;
                    for (int j = 0; j < fill; j++) {
                        float tmp = 0;
                        int ptrq = h *_head_dim;
                        int ptrk = h * _head_dim;
                        for (int k = 0; k < _head_dim; k++, ptrk++, ptrq++) {
                            tmp += cacheq[b][i](ptrq) * _cachek[b][j](ptrk);
                        }
                        tmp = std::exp(tmp / scale);
                        score[b][h].set(tmp, i, j);
                        sum += tmp;
                    }
                    for (int j = fill; j < totlen; j++) {
                        score[b][h].set(0, i, j);
                    }
                    for (int j = 0; j < fill; j++) {
                        score[b][h].set(score[b][h](i, j) / sum, i, j);
                    }
                }
            }
        }

        std::vector<std::vector<Tensor>> output(batch);
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < _num_heads; h++) {
                output[b].push_back(Tensor({seqlen, _head_dim}));
                output[b].back().zero();
                for (int k = 0; k < totlen; k++) {
                    for (int i = 0; i < seqlen; i++) {
                        for (int j = 0, ptrv=_head_dim*h; j < _head_dim; j++, ptrv++) {
                            output[b][h].add(score[b][h](i, k) * _cachev[b][k](ptrv), i, j);
                        }
                    }
                }
            }
        }

        Tensor result({batch, seqlen, _dim});
        result.zero();
        for (int b = 0; b < batch; b++) {
            for (int l = 0;  l < seqlen; l++) {
                for (int i = 0; i < _dim; i++) {
                    int ptr = 0;
                    for (int h = 0; h < _num_heads; h++) {
                        for (int j = 0; j < _head_dim; j++, ptr++) {
                            result.add(_wo(i, ptr) * output[b][h](l, j), b, l, i);
                        }
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
