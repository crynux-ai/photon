#pragma once

#include "include/backend.h"
#include "include/executor.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <map>


template <>
class Executor<BackendType::CPU> {

public:
    Executor(int batch) : batch(batch) {
    }

    int batch;

    ~Executor() {
    }

    void build() {
    }

    void forward(int func, std::vector<int> command_args, std::array<int, 3> grid_size) {
    }

    void addBuffer(int idx, void* data_ptr, size_t data_size) {
    }

    void bufferToTensor(int idx, Tensor* tensor) {
    }

private:
};

