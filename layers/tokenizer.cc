#include "tokenizer.h"

void Tokenizer::build(std::string_view path) {
    this->processor.LoadOrDie(path);
}

std::vector<int> Tokenizer::encode(std::string_view text, bool bos, bool eos) {
    std::vector<int> ids;
    if (bos) {
        ids.insert(ids.begin(), this->processor.bos_id());
    }
    if (eos) {
        ids.push_back(this->processor.eos_id());
    }
    processor.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    processor.Decode(token_ids, &text);
    return text;
}
