#include "cpu/attention.h"
#include "cpu/math_utils.h"

#include <cassert>


Tensor Attention<BackendType::CPU>::forward(
        const Tensor& input, const Tensor& rope_cost, const Tensor& rope_sint,
        int start_pos, bool mask, Tensor* residual) {
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    if (_cachek.shape().empty()) {
        _cachek = Tensor({batch, _maxlen, _dim});
        _cachev = Tensor({batch, _maxlen, _dim});
    }

    Tensor xq({batch, seqlen, _dim});

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < _dim; j++) {
                float valk = 0;
                float valv = 0;
                float valq = 0;
                for (int k = 0; k < _dim; k++) {
                    valk += input(b, i, k) * _wk(j, k);
                    valv += input(b, i, k) * _wv(j, k);
                    valq += input(b, i, k) * _wq(j, k);
                }
                _cachek.set(valk, b, start_pos + i, j);
                _cachev.set(valv, b, start_pos + i, j);
                xq.set(valq, b, i, j);
            }
        }
    }

    apply_rotary_emb<BackendType::CPU>(&xq, &_cachek, rope_cost, rope_sint, start_pos, seqlen);

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
                        tmp += xq(b, i, ptrq) * _cachek(b, j, ptrk);
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
                        output[b][h].add(score[b][h](i, k) * _cachev(b, k, ptrv), i, j);
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
