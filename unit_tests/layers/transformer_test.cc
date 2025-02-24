#include "include/backend.h"
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


    ModelArgs args = {
        .dim = dim,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .vocab_size = vocab_size,
        .multiple_of = multiple_of,
        .max_seq_len = maxseqlen,
    };
    Transformer<CURRENT_BACKEND> layer(args);
    layer.build(loader.Read(layer.size()));
    
    std::vector<std::vector<std::vector<int>>> inputs;
    std::vector<Tensor> outputs;
    for (int i = 0; i < num_cases; i++) {
        std::vector<std::vector<int>> tokens(shapes[i].first);
        for (int j = 0; j < shapes[i].first; j++) {
            for (int k = 0; k < shapes[i].second; k++) {
                tokens[j].push_back(loader.ReadInt());
            }
        }
        inputs.push_back(tokens);

        Tensor output;
        output.build(loader.Read(shapes[i].first * shapes[i].second * 4 * vocab_size + 16));
        outputs.push_back(std::move(output));
    }

    int start_pos = 0;
    for (int i = 0; i < num_cases; i++) {
        auto start = high_resolution_clock::now();
        auto result = layer.forward(inputs[i], start_pos);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Time: " << duration.count() << " microseconds" << std::endl;

        start_pos += inputs[i][0].size();
        EXPECT_EQ(result.eq(outputs[i]), true);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
