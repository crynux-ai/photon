#pragma once

#include "include/backend.h"
#include "include/rope.h"
#include "include/params.h"
#include "schema/tensor.h"
#include <cmath>


template <BackendType backend>
class Attention {

public:
    virtual ~Attention() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual void forward(const RunParams& param) = 0;

protected:
    Attention(int dim, int num_heads) = default;
};

template <> class Attention<BackendType::CPU>;

template <> class Attention<BackendType::METAL>;

#ifdef PHOTON_METAL
#include "metal/attention.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/attention.h"
    #endif
#endif
