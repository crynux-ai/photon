struct RunParams {
    uint batch;
    uint seq_len;
    uint max_seq_len;
    uint start_pos;

    uint dim;
    uint num_heads;
    uint actual_hidden_dim;
    uint head_dim;
    uint num_complex;
    uint vocab_size;

    int mask;
    int residual;

    float norm_eps;
};
