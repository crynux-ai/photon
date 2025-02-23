#include "layers/tokenizer.h"

#include <iostream>
void Tokenizer::build(std::string_view path) {
    processor.LoadOrDie(path);
}

std::vector<int> Tokenizer::encode(std::string_view text, bool bos, bool eos) {
    std::vector<int> ids;
    processor.Encode(text, &ids);
    if (bos) {
        ids.insert(ids.begin(), processor.bos_id());
    }
    if (eos) {
        ids.push_back(processor.eos_id());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    processor.Decode(token_ids, &text);
    return text;
}
