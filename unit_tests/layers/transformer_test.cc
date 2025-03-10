#include "include/backend.h"
#include "include/executor.h"
#include "include/profiler.h"
#include "include/transformer.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <gtest/gtest.h>

using namespace std::chrono;

TEST(Transformer, TransformerTest) {
    Loader loader("unit_tests/testdata/transformer.dat");
    int head_dim = loader.ReadInt();
    int maxseqlen = loader.ReadInt();
    int dim = loader.ReadInt();
    int num_layers = loader.ReadInt();
    int num_heads = loader.ReadInt();
    int multiple_of = loader.ReadInt();
    int vocab_size = loader.ReadInt();
    int num_cases = loader.ReadInt();
    std::vector<std::pair<int, int>> shapes;
    for (int i = 0; i < num_cases; i++) {
        int d1 = loader.ReadInt();
        int d2 = loader.ReadInt();
        shapes.push_back({d1, d2});
    }

METAL_ARC_BEGIN
    ModelArgs args = {
        .dim = dim,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .vocab_size = vocab_size,
        .multiple_of = multiple_of,
        .max_seq_len = maxseqlen,
    };

    RunParams params = {
        .batch = 3,
        .seq_len = 0,
        .max_seq_len = maxseqlen,
        .start_pos = 0,
        .dim = dim,
        .actual_hidden_dim = FFNSwiGLU<CURRENT_BACKEND>::calc_hidden_dim(dim*4, multiple_of),
        .num_heads = num_heads,
        .head_dim = dim / num_heads,
        .num_complex = dim / num_heads /2,
        .vocab_size = vocab_size,
        .mask = false,
        .residual = false,
    };


    auto executor = std::make_shared<Executor<CURRENT_BACKEND>>(/*batch=*/3);
    executor->build();
    Transformer<CURRENT_BACKEND> layer(args, executor);
    layer.build(loader.Read(layer.size()));
    
    std::vector<std::unique_ptr<Tensor>> inputs;
    std::vector<Tensor> outputs;
    for (int i = 0; i < num_cases; i++) {
        std::unique_ptr<Tensor> tokens = std::make_unique<Tensor>(std::vector{shapes[i].first, shapes[i].second});
        for (int j = 0; j < shapes[i].first; j++) {
            for (int k = 0; k < shapes[i].second; k++) {
                tokens->set(float(loader.ReadInt()), j, k);
            }
        }
        inputs.push_back(std::move(tokens));

        Tensor output;
        output.build(loader.Read(shapes[i].first * shapes[i].second * 4 * vocab_size + 16));
        outputs.push_back(std::move(output));
    }

    int start_pos = 0;
    for (int i = 0; i < num_cases; i++) {
        int batch = inputs[i]->shape()[0];
        int seqlen = inputs[i]->shape()[1];
        params.seq_len = seqlen;
        executor->addBuffer(layer.obj_id, Transformer_INPUT, *inputs[i]);
        layer.alloc_shared_buffer(params);

        auto start = high_resolution_clock::now();
        int repeat_cnt = 1;
        for (int j=0; j < repeat_cnt; j++) {
            layer.forward(params);
            executor->waitUntilCompleted();
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        std::cout << "Time: " << duration.count() / repeat_cnt << " ns" << std::endl;
        PROFILE_PRINT

        params.start_pos += params.seq_len;
        auto result = executor->bufferToTensor(layer.obj_id, Transformer_RESULT, {batch, seqlen, vocab_size});
        EXPECT_EQ(result->eq(outputs[i], true), true);
    }
METAL_ARC_END
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
