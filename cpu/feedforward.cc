#include "cpu/feedforward.h"
#include "layers/math_utils.h"

#include <cassert>


Tensor FFNSwiGLU<BackendType::CPU>::forward(const Tensor& input, Tensor* residual) {
    assert(input.shape().size() == 3);
    int batch = input.shape()[0];
    int seqlen = input.shape()[1];
    assert(input.shape()[2] == _dim);

    Tensor r1({batch, seqlen, _hidden_dim});
    r1.zero();
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seqlen; i++) {
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
