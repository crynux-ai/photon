#include "include/backend.h"
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
    
    Tensor wq, wk, wv, wo;
    int tensor_size = dim * dim * 4 + 12;
    Attention<CURRENT_BACKEND> layer(dim, num_head, maxseqlen);
    layer.build(loader.Read((dim * dim * 4 + 12) * 4));

    Tensor x1, y1, x2, y2, x3, y3, p1, p2, p3;
    x1.build(loader.Read(3*7*256*4 + 16));
    y1.build(loader.Read(3*7*256*4 + 16));
    x2.build(loader.Read(3*3*256*4 + 16));
    y2.build(loader.Read(3*3*256*4 + 16));
    x3.build(loader.Read(3*2*256*4 + 16));
    y3.build(loader.Read(3*2*256*4 + 16));

    p1 = layer.forward(x1, rope_cost, rope_sint, 0, true);
    p2 = layer.forward(x2, rope_cost, rope_sint, 7, true);
    p3 = layer.forward(x3, rope_cost, rope_sint, 10, false);
    
    EXPECT_EQ(p1.eq(y1, true), true);
    EXPECT_EQ(p2.eq(y2, true), true);
    EXPECT_EQ(p3.eq(y3, true), true);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
