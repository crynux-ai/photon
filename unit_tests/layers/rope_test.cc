#include "layers/rope.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>


TEST(RopeTest, RopeTest) {
    Loader loader("unit_tests/testdata/rope.dat");
    int batch = loader.ReadInt();
    int seqlen = loader.ReadInt();
    int num_head = loader.ReadInt();
    int head_dim = loader.ReadInt();
    
    Tensor xq, xk, pq, pk;
    int tensor_size = batch * seqlen * num_head * head_dim * 4 + 16;

    xq.build(loader.Read(tensor_size));
    xk.build(loader.Read(tensor_size));
    pq.build(loader.Read(tensor_size));
    pk.build(loader.Read(tensor_size));

    auto freqs_ics = precompute_freqs_cis(head_dim, seqlen, 10000.0);
    apply_rotary_emb(&xq, &xk, freqs_ics.first, freqs_ics.second);
    EXPECT_EQ(xq.eq(pq), true);
    EXPECT_EQ(xk.eq(pk), true);
}
