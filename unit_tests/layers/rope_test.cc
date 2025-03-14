#include "include/rope.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>


TEST(RopeTest, RopeTest) {
    Loader loader("unit_tests/testdata/rope.dat");
    int batch = loader.ReadInt();
    int maxseqlen = loader.ReadInt();
    int startpos = loader.ReadInt();
    int seqlen = loader.ReadInt();
    int num_head = loader.ReadInt();
    int head_dim = loader.ReadInt();
    
    Tensor xq, xk, pq, pk;
    int tensor_size = batch * seqlen * num_head * head_dim * 4 + 16;

    xq.build(loader.Read(tensor_size));
    xk.build(loader.Read(tensor_size));
    pq.build(loader.Read(tensor_size));
    pk.build(loader.Read(tensor_size));
    int dim = xq.shape()[2];

    Tensor vxq({batch, seqlen, dim});
    Tensor vxk({batch, maxseqlen, dim});
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seqlen; j++) {
            for (int k = 0; k < dim; k++) {
                vxq.set(xq(i, j, k), i, j, k);
                vxk.set(xk(i, j, k), i, j + startpos, k);
            }
        }
    }

    Tensor cost({maxseqlen, head_dim / 2});
    Tensor sint({maxseqlen, head_dim / 2});
    precompute_freqs_cis(head_dim, maxseqlen, 10000.0, &cost, &sint);
    apply_rotary_emb<CURRENT_BACKEND>(&vxq, &vxk, cost, sint, startpos, seqlen);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seqlen; j++) {
            for (int k = 0; k < xq.shape()[2]; k++) {
                xq.set(vxq(i, j, k), i, j, k);
                xk.set(vxk(i, startpos + j, k), i, j, k);
            }
        }
    }

    EXPECT_EQ(xq.eq(pq, true), true);
    EXPECT_EQ(xk.eq(pk, true), true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
