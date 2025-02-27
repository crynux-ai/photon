#include "include/rope.h"

void precompute_freqs_cis(
        int head_dim, int max_seqlen, float theta,
        Tensor* cost, Tensor* sint) {
    assert(head_dim % 2 == 0);
    int size = head_dim / 2;

    assert(cost->shape() == sint->shape());
    assert(cost->shape() == std::vector({max_seqlen, size}));

    std::vector<float> freqs(size);
    for (int i = 0; i < size; i++) {
        freqs[i] = 1.0 / std::pow(theta, float(i + i) / float(head_dim));
    }

    for (int i = 0; i < max_seqlen; i++) {
        for (int j = 0; j < size; j++) {
            cost->set(cos(i * freqs[j]), i, j);
            sint->set(sin(i * freqs[j]), i, j);
        }
    }
}
