#pragma once

#include <cmath>


inline float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}
