#pragma once

enum class BackendType {
    CPU,
    CUDA,
    METAL
};

#ifdef PHOTON_METAL
    #define CURRENT_BACKEND BackendType::METAL
#else
    #ifdef PHOTON_CUDA
        #define CURRENT_BACKEND BackendType::CUDA
    #else
        #define CURRENT_BACKEND BackendType::CPU
    #endif
#endif
