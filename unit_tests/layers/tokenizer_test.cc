#include "layers/tokenizer.h"

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

    std::ifstream data("testdata/sentencepiece.dat");
    if (!data) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    std::vector<std::vector<int>> tokens;
    std::vector<std::string> texts;
    std::string line;

    // Process test data
    while (std::getline(data, line)) {
        texts.push_back(line);

        std::vector<int> item;
        std::getline(data, line);
        std::istringstream iss(line);
        std::string token;

        // Use comma as delimiter
        while (std::getline(iss, token, ',')) {
            // Remove possible spaces
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            try {
                int num = std::stoi(token);
                item.push_back(num);
            } catch (const std::exception& e) {
                std::cerr << "Conversion error: " << token << " is not a valid integer." << std::endl;
            }
        }
        tokens.push_back(item);
    }

    for (int i = 0; i < texts.size(); i++) {
        EXPECT_EQ(tokens[i], tokenizer.encode(texts[i], true, true));
        EXPECT_EQ(texts[i], tokenizer.decode(tokens[i]));
    }
}
