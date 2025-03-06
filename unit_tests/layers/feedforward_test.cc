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

METAL_ARC_BEGIN
    auto executor = std::make_shared<Executor<CURRENT_BACKEND>>(3);
    executor->build();
    FFNSwiGLU<CURRENT_BACKEND> ffn(dim, hidden_dim, multiple_of, executor);
    ffn.build(loader.Read(ffn.size()));

    int tensor_size = 3 * 7 * 256 * 4 + 16;
    Tensor x;
    x.build(loader.Read(tensor_size));
    Tensor y;
    y.build(loader.Read(tensor_size));

    tensor_size =  3*7*256*4;
    executor->addBuffer(ffn.obj_id, FFNSwiGLU_INPUT, x._value.get(), tensor_size);
    ffn.forward(7, false);
    Tensor z({3, 7, 256});
    executor->bufferToTensor(ffn.obj_id, FFNSwiGLU_RESULT, &z);
    EXPECT_EQ(z.eq(y), true);
METAL_ARC_END
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
