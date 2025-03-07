#pragma once

#include "include/backend.h"
#include "include/params.h"
#include "schema/tensor.h"
#include <cmath>


template <BackendType backend>
class Transformer {

public:
    virtual ~Transformer() = default;

    virtual size_t size() = 0;

    virtual void build(std::string_view content) = 0;

    virtual void forward(const RunParams& param) = 0;

protected:
    Transformer(const ModelArgs& args) = default;

};


template <> class Transformer<BackendType::CPU>;

template <> class Transformer<BackendType::METAL>;

#ifdef PHOTON_METAL
#include "metal/transformer.h"
#else
    #ifdef PHOTON_CUDA
    #else
        #include "cpu/transformer.h"
    #endif
#endif

