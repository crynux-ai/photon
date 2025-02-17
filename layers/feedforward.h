#ifndef LAYERS_FEEDFORWARD_H
#define LAYERS_FEEDFORWARD_H

#include "schema/loader.h"
#include "schema/tensor.h"
#include "layers/math.h"


class FFNSwiGLU {

public:
    FFNSwiGLU(int dim, int hidden_dim, int multiple_of) {
        _dim = dim;
        _hidden_dim = multiple_of * ((2 * hidden_dim / 3 + multiple_of - 1) / multiple_of);
        _w1 = Tensor({_hidden_dim, _dim});
        _w2 = Tensor({_dim, _hidden_dim});
        _w3 = Tensor({_hidden_dim, _dim});
        
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

    Tensor forward(const Tensor& input) {
        int batch = input.shape()[0];
        Tensor result(input.shape());

        Tensor r1({batch, _hidden_dim});
        r1.zero();
        result.zero();
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < _hidden_dim; j++) {
                float val1 = 0;
                float val2 = 0;
                for (int k = 0; k < _dim; k++) {
                    val1 += input(i, k) * _w1(j, k);
                    val2 += input(i, k) * _w3(j, k);
                }
                r1.set(val1 * val2 * sigmoid(val1), i, j);
            }
        }

        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < _dim; j++) {
                for (int k = 0; k < _hidden_dim; k++) {
                    result.add(r1(i, k) * _w2(j, k), i, j);
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