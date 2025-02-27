#include "include/rope.h"

#include <cmath>
#include <cassert>
#include <iostream>

template<>
void apply_rotary_emb<BackendType::CPU>(
        Tensor* xq,
        Tensor* cachek,
        const Tensor& cost,
        const Tensor& sint,
        int start_pos,
        int seqlen) {
    int batch = xq->shape()[0];
    int dim = xq->shape()[2];
    int num_complex = cost.shape()[1];
    int head_dim = num_complex * 2;
    int num_head = dim / head_dim;
    assert(dim % head_dim == 0);

    for (int b = 0; b < batch; b++) {
        int i = 0;
        int ik = start_pos;
        for (; i < seqlen; i++, ik++) {
            int p0 = 0;
            int p1 = 1;
            for (int j = 0; j < num_head; j++) {
                for (int d = 0; d < num_complex; d++, p0+=2, p1+=2) {
                    float xq0 = (*xq)(b, i, p0);
                    float xq1 = (*xq)(b, i, p1);
                    xq->set(xq0 * cost(ik, d) - xq1 * sint(ik, d), b, i, p0);
                    xq->set(xq1 * cost(ik, d) + xq0 * sint(ik, d), b, i, p1);

                    float xk0 = (*cachek)(b, ik, p0);
                    float xk1 = (*cachek)(b, ik, p1);
                    cachek->set(xk0 * cost(ik, d) - xk1 * sint(ik, d), b, ik, p0);
                    cachek->set(xk1 * cost(ik, d) + xk0 * sint(ik, d), b, ik, p1);
                }
            }
        }
    }
}
