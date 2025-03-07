#include "include/backend.h"
#include "include/feedforward.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>


TEST(FeedForwardTest, FFNSwiGLUTest) {
    Loader loader("unit_tests/testdata/feedforward.dat");
    int dim = loader.ReadInt();
    int hidden_dim = loader.ReadInt();
    int multiple_of = loader.ReadInt();

    RunParams param = {
        .batch = 3,
        .seq_len = 7,
        .start_pos = 0,
        .dim = dim,
        .actual_hidden_dim = FFNSwiGLU<CURRENT_BACKEND>::calc_hidden_dim(hidden_dim, multiple_of),
        .residual =false,
    };

METAL_ARC_BEGIN
    auto executor = std::make_shared<Executor<CURRENT_BACKEND>>(3);
    executor->build();
    FFNSwiGLU<CURRENT_BACKEND> ffn(dim, hidden_dim, multiple_of, executor);
    ffn.build(loader.Read(ffn.size()));

    int tensor_bytes = 3 * 7 * 256 * 4 + 16;
    Tensor x;
    x.build(loader.Read(tensor_bytes));
    Tensor y;
    y.build(loader.Read(tensor_bytes));

    executor->addBuffer(ffn.obj_id, FFNSwiGLU_INPUT, x);
    ffn.alloc_shared_buffer(param);
    ffn.forward(param);
    auto z = executor->bufferToTensor(ffn.obj_id, FFNSwiGLU_RESULT, {3, 7, 256});
    EXPECT_EQ(z->eq(y, true), true);
METAL_ARC_END
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
