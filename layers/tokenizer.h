#ifndef LAYERS_TOKENIZER_H
#define LAYERS_TOKENIZER_H

#include "src/sentencepiece_processor.h"

#include <string_view>
#include <vector>

class Tokenizer {

public:
    Tokenizer() {}
    void build(std::string_view path);
    std::vector<int> encode(std::string_view text);
    std::string decode(const std::vector<int>& token_ids);
private:
    sentencepiece::SentencePieceProcessor processor;
};

#endif // LAYERS_TOKENIZER_H