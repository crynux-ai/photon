#ifndef LAYERS_FEEDFORWARD_H
#define LAYERS_FEEDFORWARD_H

#include "schema/loader.h"
#include "schema/tensor.h"
#include "layers/math.h"

#include <cassert>

class FFNSwiGLU {

public:
    FFNSwiGLU(int dim, int hidden_dim, int multiple_of) {
        _dim = dim;
        _hidden_dim = multiple_of * ((2 * hidden_dim / 3 + multiple_of - 1) / multiple_of);
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
    }

    Tensor forward(const Tensor& input, bool residual=false) {
        assert(input.shape().size() == 3);
        int batch = input.shape()[0];
        int seqlen = input.shape()[1];
        assert(input.shape()[2] == _dim);

        Tensor r1({batch, _dim, _hidden_dim});
        r1.zero();
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < _dim; i++) {
                for (int j = 0; j < _hidden_dim; j++) {
                    float val1 = 0;
                    float val2 = 0;
                    for (int k = 0; k < _dim; k++) {
                        val1 += input(b, i, k) * _w1(j, k);
                        val2 += input(b, i, k) * _w3(j, k);
                    }
                    r1.set(val1 * val2 * sigmoid(val1), b, i, j);
                }
            }
        }

        Tensor result(input.shape());
        result.zero();
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqlen; i++) {
                for (int j = 0; j < _dim; j++) {
                    if (residual) {
                        result.add(input(b, i, j));
                    }
                    for (int k = 0; k < _hidden_dim; k++) {
                        result.add(r1(b, i, k) * _w2(j, k), b, i, j);
                    }
                }
            }
        }
       
        return result;
    }

private:
    Tensor _w1;
    Tensor _w2;
    Tensor _w3;
    int _dim;
    int _hidden_dim;
};

#endif // LAYERS_FEEDFORWARD_H