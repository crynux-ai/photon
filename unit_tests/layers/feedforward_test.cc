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

    FFNSwiGLU<CURRENT_BACKEND> ffn(dim, hidden_dim, multiple_of);
    ffn.build(loader.Read(ffn.size()));

    int tensor_size = 3 * 7 * 256 * 4 + 16;
    Tensor x;
    x.build(loader.Read(tensor_size));
    Tensor y;
    y.build(loader.Read(tensor_size));

    auto z = ffn.forward(x);
    EXPECT_EQ(z.eq(y), true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
