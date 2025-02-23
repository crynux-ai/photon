#include "schema/tokenizer.h"
#include "schema/loader.h"

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>


TEST(TokenizerTest, TokenizerInfo) {
    Tokenizer tokenizer;
    tokenizer.build("models/tokenizer.model");
    EXPECT_EQ(tokenizer.processor.bos_id(), 1);
    EXPECT_EQ(tokenizer.processor.eos_id(), 2);
    EXPECT_EQ(tokenizer.processor.pad_id(), -1);
    EXPECT_EQ(tokenizer.processor.GetPieceSize(), 32000);
}

TEST(TokenizerTest, EncodeDecode) {
    Tokenizer tokenizer;
    tokenizer.build("models/tokenizer.model");
    auto ids = tokenizer.encode("hello world", true, true);
    auto text = tokenizer.decode(ids);
    EXPECT_EQ(text, "hello world") << text;

    ids = tokenizer.encode("", true, true);
    text = tokenizer.decode(ids);
    EXPECT_EQ(text, "") << text;

    ids = tokenizer.encode("  ", true, true);
    text = tokenizer.decode(ids);
    EXPECT_EQ(text, "  ") << text;
}

TEST(TokenizerTest, MatchPython) {
    Tokenizer tokenizer;
    tokenizer.build("models/tokenizer.model");
    Loader loader("unit_tests/testdata/sentencepiece.dat");
    int num_cases = loader.ReadInt();

    std::vector<std::vector<int>> tokens(num_cases);
    std::vector<std::string> texts;
    for (int i = 0; i < num_cases; i++) {
        int len = loader.ReadInt();
        std::string str(loader.Read(len));
        texts.push_back(str);
        len = loader.ReadInt();
        for (int j = 0; j < len; j++) {
            tokens[i].push_back(loader.ReadInt());
        }
    }
    for (int i = 0; i < texts.size(); i++) {
        EXPECT_EQ(tokens[i], tokenizer.encode(texts[i], true, true));
        EXPECT_EQ(texts[i], tokenizer.decode(tokens[i]));
    }
}
