#ifndef SCHEMA_LAYER_H
#define SCHEMA_LAYER_H

#include <fstream>
#include <string_view>
#include <iostream>

class Loader {

public:
    Loader(std::string_view filepath) {
        _file = std::ifstream(filepath, std::ios::binary | std::ios::ate);
        _size = _file.tellg();
        _file.seekg(0, std::ios::beg);
        _buffer = NULL;
    }

    ~Loader() {
        _file.close();
        if (_buffer) {
            delete[] _buffer;
        }
    }

    size_t size() {
        return _size;
    }

    std::string_view Read(size_t size) {
        if (_buffer) {
            delete[] _buffer;
        }

        _buffer = new char[size];
        if (!_file.read(_buffer, size)) {
            delete[] _buffer;
            throw std::runtime_error("Error reading file");
        }
        return {_buffer, size};
    }

    int ReadInt() {
        auto data = Read(4);
        int res;
        std::memcpy(&res, data.data(), 4);
        return res;
    }

private:
    std::ifstream _file;
    std::streamsize _size;
    char* _buffer;
};

#endif  // SCHEMA_MODEL_H
