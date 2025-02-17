#ifndef SCHEMA_TENSOR_H
#define SCHEMA_TENSOR_H


#include <string_view>
#include <vector>

class Tensor {

public:
    Tensor() {
        _cnt = 0;
        _value = NULL;
    }

    Tensor(const std::vector<int>& shape) {
        _shape = shape;
        _cnt = 1;
        for (int x : shape) {
            _cnt *= x;
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
        

        _cnt = 1;
        for (int i = 0; i < dim; i++) {
            std::memcpy(&_shape[i], ptr, 4);
            ptr += 4;
            _cnt *= _shape[i];
        }

        _value = new float[_cnt];
        std::memcpy(_value, ptr, 4 * _cnt);
    }

    bool eq(const Tensor& other) {
        if (_shape != other.shape()) {
            return false;
        }
        for (int i = 0; i < _cnt; i++) {
            if (std::fabs(_value[i] - other._value[i]) > 1e-5) {
                return false;
            }
        }
        return true;
    }

protected:
    std::vector<int> _shape;
    int _cnt;
    float* _value;
    
};

#endif // SCHEMA_TENSOR_H