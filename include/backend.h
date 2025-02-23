#pragma once

enum class BackendType {
    CPU,
    CUDA,
    METAL
};

#ifdef METAL
    #define CURRENT_BACKEND BackendType::METAL
#else
    #ifdef CUDA
        #define CURRENT_BACKEND BackendType::CUDA
    #else
        #define CURRENT_BACKEND BackendType::CPU
    #endif
#endif
