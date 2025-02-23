#ifndef LAYERS_MATH_H
#define LAYERS_MATH_H

#include <cmath>


inline float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

#endif  // LAYERS_MATH_H