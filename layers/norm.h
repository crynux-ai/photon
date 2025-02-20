#ifndef LAYERS_NORM_H
#define LAYERS_NORM_H

#include "schema/tensor.h"

#include <cmath>


Tensor RMSNorm(const Tensor& x, float norm_eps) {
    assert (x.shape().size() == 3);
    auto batch = x.shape()[0];
    auto seqlen = x.shape()[1];
    auto dim = x.shape()[2];

    Tensor result(x.shape());
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seqlen; j++) {
            float sum = 0;
            for (int k = 0; k < dim; k++) {
                float tmp = x(i, j, k);
                sum += tmp * tmp;
            }
            sum /= dim;
            sum += norm_eps;
            sum = 1.0f / std::sqrt(sum);
            for (int k = 0; k < dim; k++) {
                result.set(x(i, j, k) * sum, i, j, k);
            }
        }
    }
    return result;
}

#endif  // LAYERS_NORM_H


