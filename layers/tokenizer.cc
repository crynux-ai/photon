#include "tokenizer.h"

#include <stdexcept>

void Tokenizer::build(std::string_view path) {
    const auto status = this->processor.Load(path);
    if (!status.ok()) {
        throw std::invalid_argument(status.ToString());
    }
}

std::vector<int> Tokenizer::encode(std::string_view text) {
    std::vector<int> ids;
    processor.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    processor.Decode(token_ids, &text);
    return text;
}
