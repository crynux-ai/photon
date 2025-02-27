#pragma once

#include "include/backend.h"
#include "schema/tensor.h"

#include <cmath>
#include <cassert>
#include <iostream>

void precompute_freqs_cis(
        int head_dim, int max_seqlen, float theta,
        Tensor* cost, Tensor* sint);

template <BackendType backend>
void apply_rotary_emb(
        Tensor* xq,
        Tensor* cachek,
        const Tensor& cost,
        const Tensor& sint,
        int start_pos,
        int seqlen);
