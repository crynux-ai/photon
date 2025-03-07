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


enum TensorEnum {
    Attention_INPUT = 0,            // int [batch, seqlen, dim]
    Attention_CACHE_K = 1,          // float [batch, maxseqlen, dim]
    Attention_CACHE_V = 2,          // float [batch, maxseqlen, dim]
    Attention_WEIGHT_Q = 3,         // float [dim, dim]
    Attention_WEIGHT_K = 4,         // float [dim, dim]
    Attention_WEIGHT_V = 5,         // float [dim, dim]
    Attention_WEIGHT_O = 6,         // float [dim, dim]
    Attention_XQ = 7,               // float [batch, seqlen, dim]
    Attention_SCORE = 8,            // float [batch, seqlen, maxseqlen, num_head]
    Attention_OUTPUT = 9,           // float [batch, seqlen, dim]
    Attention_RESULT = 10,          // float [batch, seqlen, dim]
    Attention_RESIDUAL = 11,        // float [batch, seqlen, dim]

    Rope_COST = 15,                 // float [maxseqlen, head_dim / 2]
    Rope_SINT = 16,                 // float [maxseqlen, head_dim / 2]

    FFNSwiGLU_INPUT = 20,           // float [batch, seqlen, dim]
    FFNSwiGLU_W1 = 21,              // float [actual_hidden_dim, dim]
    FFNSwiGLU_W2 = 22,              // float [dim, actual_hidden_dim]
    FFNSwiGLU_W3 = 23,              // float [actual_hidden_dim, dim]
    FFNSwiGLU_RESIDUAL = 24,        // float [batch, seqlen, dim]
    FFNSwiGLU_RESULT = 25,          // float [batch, seqlen, dim]
    FFNSwiGLU_HIDDEN_OUTPUT = 26,   // float [batch, seqlen, actual_dim]

    Transformer_INPUT = 30,             // float [batch, seqlen]
    Transformer_EMBEDDING_TABLE = 31,   // float [vocab, dim]
    Transformer_WEIGHT_O = 32,          // float [vocab, dim]
    Transformer_INPUT_EMBEDDING = 33,   // float [batch, seqlen, dim]
    Transformer_OUTPUT = 35,            // float [batch, seqlen, dim]
    Transformer_RESULT = 36,            // float [batch, seqlen, vocab]
    Transformer_INPUT_NORMS = 37,       // float [batch, seqlen, dim]
};

static int MAX_TENSOR_ENUM = 40;
