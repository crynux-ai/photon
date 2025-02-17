#include "layers/feedforward.h"
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
    int layer_size = (256 * 684 * 4 + 12) * 3;

    FFNSwiGLU ffn(dim, hidden_dim, multiple_of);
    ffn.build(loader.Read(layer_size));
    

    int tensor_size = 256 * 4 + 12;
    Tensor x({1, dim});
    x.build(loader.Read(tensor_size));
    Tensor y({1, dim});
    y.build(loader.Read(tensor_size));

    auto z = ffn.forward(x);
    EXPECT_EQ(z.eq(y), true);
}

