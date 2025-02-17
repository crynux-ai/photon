#ifndef SCHEMA_TENSOR_H
#define SCHEMA_TENSOR_H


#include <string_view>
#include <vector>
#include <iostream>

class Tensor {

public:
    Tensor() {
        _cnt = 0;
        _value = NULL;
    }

    Tensor(const std::vector<int>& shape) {
        _shape = shape;
        _cnt = 1;
        _base.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; i--) {
            _base[i] = _cnt;
            _cnt *= shape[i];
        }
        _value = new float[_cnt];
    }

    ~Tensor() {
        if (_value) {
            delete[] _value;
        }
    }

    void zero() {
        memset(_value, 0, sizeof(float) * _cnt);
    }
    
    const std::vector<int>& shape() const {
        return _shape;
    }

    // TODO: use mmap
    void build(std::string_view data) {
        int dim;
        auto ptr = data.data();
        std::memcpy(&dim, ptr, 4);
        ptr += 4;
        _shape.resize(dim);
        _base.resize(dim);
        

        for (int i = 0; i < dim; i++) {
            std::memcpy(&_shape[i], ptr, 4);
            ptr += 4;
        }

        _cnt = 1;
        for (int i = _shape.size() - 1; i >= 0; i--) {
            _base[i] = _cnt;
            _cnt *= _shape[i];
        }

        _value = new float[_cnt];
        std::memcpy(_value, ptr, 4 * _cnt);
    }

    bool eq(const Tensor& other, bool check_all=false) {
        if (_shape != other.shape()) {
            return false;
        }
        int cnt = _cnt;
        for (int i = 0; i < _cnt; i++) {
            if (std::fabs(_value[i] - other._value[i]) > 1e-5) {
                if (check_all) {
                    cnt--;
                } else {
                    return false;
                }
            }
        }
        if (cnt < _cnt) {
            std::cerr << cnt << " / " << _cnt << " matches" << std::endl;
        }
        return cnt == _cnt;
    }

    template <typename... Args>
    float operator()(Args... args) const {
        if (sizeof...(args) != _shape.size()) {
            throw std::invalid_argument("Invalid access to tensor");
        }
        int index = 0;
        int ptr = 0;
        ((index += _base[ptr++] * args), ...);  
        return _value[index];
    }

    template <typename... Args>
    void set(float val, Args... args) {
        if (sizeof...(args) != _shape.size()) {
            throw std::invalid_argument("Invalid access to tensor");
        }
        int index = 0;
        int ptr = 0;
        ((index += _base[ptr++] * args), ...);  
        _value[index] = val;
    }

    template <typename... Args>
    void add(float val, Args... args) {
        if (sizeof...(args) != _shape.size()) {
            throw std::invalid_argument("Invalid access to tensor");
        }
        int index = 0;
        int ptr = 0;
        ((index += _base[ptr++] * args), ...);  
        _value[index] += val;
    }


protected:
    std::vector<int> _shape;
    std::vector<int> _base;
    int _cnt;
    float* _value;
    
};

#endif // SCHEMA_TENSOR_H