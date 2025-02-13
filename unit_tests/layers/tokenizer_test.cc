#include "layers/tokenizer.h"

#include <gtest/gtest.h>

#include <iostream>

TEST(TokenizerTest, EncodeDecode) {
    Tokenizer tokenizer;
    tokenizer.build("models/tokenizer.model");
    auto ids = tokenizer.encode("hello world");
    auto text = tokenizer.decode(ids);
    EXPECT_EQ(text, "hello world") << text;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
