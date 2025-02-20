#ifndef LAYERS_ROPE_H
#define LAYERS_ROPE_H

#include "schema/tensor.h"

#include <cmath>
#include <cassert>
#include <iostream>

using FreqMatrix = std::vector<std::vector<float>>;

std::pair<FreqMatrix, FreqMatrix> precompute_freqs_cis(int head_dim, int max_seqlen, float theta) {
    assert(head_dim % 2 == 0);
    int size = head_dim / 2;
    std::vector<float> freqs(size);
    for (int i = 0; i < size; i++) {
        freqs[i] = 1.0 / std::pow(theta, float(i + i) / float(head_dim));
    }

    FreqMatrix sint(max_seqlen, std::vector<float>(size));
    FreqMatrix cost(max_seqlen, std::vector<float>(size));
    for (int i = 0; i < max_seqlen; i++) {
        for (int j = 0; j < size; j++) {
            cost[i][j] = cos(i * freqs[j]);
            sint[i][j] = sin(i * freqs[j]);
        }
    }
    return {cost, sint};
}

void apply_rotary_emb(
        std::vector<std::vector<Tensor>>* xq,
        std::vector<std::vector<Tensor>>* cachek,
        const FreqMatrix& cost,
        const FreqMatrix& sint,
        int start_pos,
        int seqlen) {
    int batch = xq->size();
    int dim = (*xq)[0][0].shape()[0];
    int num_complex = cost[0].size();
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
                    float xq0 = (*xq)[b][i](p0);
                    float xq1 = (*xq)[b][i](p1);
                    (*xq)[b][i].set(xq0 * cost[ik][d] - xq1 * sint[ik][d], p0);
                    (*xq)[b][i].set(xq1 * cost[ik][d] + xq0 * sint[ik][d], p1);

                    float xk0 = (*cachek)[b][ik](p0);
                    float xk1 = (*cachek)[b][ik](p1);
                    (*cachek)[b][ik].set(xk0 * cost[ik][d] - xk1 * sint[ik][d], p0);
                    (*cachek)[b][ik].set(xk1 * cost[ik][d] + xk0 * sint[ik][d], p1);
                }
            }
        }
    }
}

#endif  // LAYERS_ROPE_H