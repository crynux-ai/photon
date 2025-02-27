#pragma once
#include "schema/tensor.h"

#include <cmath>
#include <cassert>
#include <iostream>

using FreqMatrix = std::vector<std::vector<float>>;

std::pair<FreqMatrix, FreqMatrix> precompute_freqs_cis(int head_dim, int max_seqlen, float theta);

void apply_rotary_emb(
        std::vector<std::vector<Tensor>>* xq,
        std::vector<std::vector<Tensor>>* cachek,
        const FreqMatrix& cost,
        const FreqMatrix& sint,
        int start_pos,
        int seqlen);
