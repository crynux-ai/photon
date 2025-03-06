#include "include/backend.h"
#include "include/executor.h"
#include "include/attention.h"
#include "include/rope.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>


TEST(AttentionTest, AttentionTest) {
    Loader loader("unit_tests/testdata/attention.dat");
    int dim = loader.ReadInt();
    int num_head = loader.ReadInt();
    int multiple_of = loader.ReadInt();
    int head_dim = loader.ReadInt();
    int maxseqlen = loader.ReadInt();
    float theta = loader.ReadFloat();
    Tensor rope_cost({maxseqlen, head_dim / 2});
    Tensor rope_sint({maxseqlen, head_dim / 2});
    precompute_freqs_cis(head_dim, maxseqlen, theta, &rope_cost, &rope_sint);
    
METAL_ARC_BEGIN
    Tensor wq, wk, wv, wo;
    int tensor_size = dim * dim * 4 + 12;
    auto executor = std::make_shared<Executor<CURRENT_BACKEND>>(3);
    executor->build();
    Attention<CURRENT_BACKEND> layer(dim, num_head, maxseqlen, executor);
    layer.build(loader.Read((dim * dim * 4 + 12) * 4));

    size_t rope_size = maxseqlen * head_dim / 2 * sizeof(float);
    executor->addBuffer(layer.obj_id, Attention_ROPE_COST, rope_cost._value.get(), rope_size);
    executor->addBuffer(layer.obj_id, Attention_ROPE_SINT, rope_sint._value.get(), rope_size);

    Tensor x1, y1, x2, y2, x3, y3;
    x1.build(loader.Read(3*7*256*4 + 16));
    y1.build(loader.Read(3*7*256*4 + 16));
    x2.build(loader.Read(3*3*256*4 + 16));
    y2.build(loader.Read(3*3*256*4 + 16));
    x3.build(loader.Read(3*2*256*4 + 16));
    y3.build(loader.Read(3*2*256*4 + 16));

    size_t input_size = 3 * 7 * 256 * 4;
    executor->addBuffer(layer.obj_id, Attention_INPUT, x1._value.get(), input_size);
    layer.forward(7, 0, true, false);
    Tensor p1({3, 7, 256});
    executor->bufferToTensor(layer.obj_id, Attention_RESULT, &p1);

    input_size = 3 * 3 * 256 * 4;
    executor->addBuffer(layer.obj_id, Attention_INPUT, x2._value.get(), input_size);
    layer.forward(3, 7, true, false);
    Tensor p2({3, 3, 256});
    executor->bufferToTensor(layer.obj_id, Attention_RESULT, &p2);

    input_size = 3 * 2 * 256 * 4;
    executor->addBuffer(layer.obj_id, Attention_INPUT, x3._value.get(), input_size);
    layer.forward(2, 10, false, false);
    Tensor p3({3, 2, 256});
    executor->bufferToTensor(layer.obj_id, Attention_RESULT, &p3);
    
    EXPECT_EQ(p1.eq(y1, true), true);
    EXPECT_EQ(p2.eq(y2, true), true);
    EXPECT_EQ(p3.eq(y3, true), true);
METAL_ARC_END
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
