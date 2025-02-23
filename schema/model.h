#pragma once

#include <string_view>
#include <vector>

class Model {

public:
    Tokenizer() {}
    void build(std::string_view path);
    std::vector<int> encode(std::string_view text, bool bos, bool eos);
    std::string decode(const std::vector<int>& token_ids);
    sentencepiece::SentencePieceProcessor processor;
};
