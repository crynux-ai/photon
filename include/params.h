#pragma once

// Construct the model
struct ModelArgs {
    int dim;
    int num_layers;
    int num_heads;
    int vocab_size;
    int multiple_of;
    float norm_eps = 1e-5;
    int max_seq_len = 2048;
};

// Runtime params
struct RunParams {
    int batch;
    int seq_len;
    int max_seq_len;
    int start_pos;

    int dim;
    int num_heads;
    int actual_hidden_dim;
    int head_dim;
    int num_complex;
    int vocab_size;

    int mask;
    int residual;

    float norm_eps = 1e-5;
};
