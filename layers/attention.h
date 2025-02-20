#ifndef LAYERS_ATTENTION_H
#define LAYERS_ATTENTION_H

#include "schema/loader.h"
#include "schema/tensor.h"
#include "layers/math.h"
#include "layers/rope.h"
#include <cmath>


class Attention {

public:
    Attention(int dim, int num_heads) {
        _dim = dim;
        _num_heads = num_heads;
        _head_dim = dim / num_heads;
        assert(dim % num_heads == 0);

        _wq = Tensor({_dim, _dim});
        _wk = Tensor({_dim, _dim});
        _wv = Tensor({_dim, _dim});
        _wo = Tensor({_dim, _dim});
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
    }

    Tensor forward(const Tensor& input, const std::pair<FreqMatrix, FreqMatrix>& rope, int start_pos, bool mask) {
        int batch = input.shape()[0];
        int seqlen = input.shape()[1];
        if (_cachek.empty()) {
            _cachek.resize(batch);
            _cachev.resize(batch);
        }

        std::vector<std::vector<Tensor>> xq(batch);

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqlen; i++) {
                _cachek[b].push_back(Tensor({_dim}));
                _cachev[b].push_back(Tensor({_dim}));
                xq[b].push_back(Tensor({_dim}));
                _cachek[b].back().zero();
                _cachev[b].back().zero();
                xq[b].back().zero();
                for (int j = 0; j < _dim; j++) {
                    for (int k = 0; k < _dim; k++) {
                        _cachek[b].back().add(input(b, i, k) * _wk(j, k), j);
                        _cachev[b].back().add(input(b, i, k) * _wv(j, k), j);
                        xq[b].back().add(input(b, i, k) * _wq(j, k), j);
                    }
                }
            }
        }
        
        apply_rotary_emb(&xq, &_cachek, rope.first, rope.second, start_pos, seqlen);

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
                            tmp += xq[b][i](ptrq) * _cachek[b][j](ptrk);
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
                }
            }
        }
        return result;
    }

private:
    Tensor _wq;
    Tensor _wk;
    Tensor _wv;
    Tensor _wo;
    std::vector<std::vector<Tensor>> _cachek;
    std::vector<std::vector<Tensor>> _cachev;

    int _dim;
    int _num_heads;
    int _head_dim;

};

#endif // LAYERS_ATTENTION_H