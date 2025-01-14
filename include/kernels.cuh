#pragma once

#include "parameters.h"

#include <cmath>

#define CUDA_CHECK                                                 \
    {                                                              \
        cudaError_t error = cudaGetLastError();                    \
        if (error != cudaSuccess)                                  \
            printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
    }

namespace cuda
{
    void tiled_matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);
};
