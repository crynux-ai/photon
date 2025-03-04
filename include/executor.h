#pragma once

#include "include/backend.h"
#include "schema/loader.h"
#include "schema/tensor.h"

#include <cassert>

template <BackendType backend>
class Executor {

public:
    virtual ~Executor() = default;

    virtual void build() = 0;

    virtual std::unique_ptr<Tensor> tensor(int tensor_id) = 0;

    virtual void forward() = 0;

protected:
    Executor() = default;
};

template <> class Executor<BackendType::CPU>;

template <> class Executor<BackendType::METAL>;


#ifdef PHOTON_METAL
#include "metal/executor.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/executor.h"
    #endif
#endif
