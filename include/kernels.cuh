#pragma once

#include "parameters.h"

#include <cmath>

#define CUDA_CHECK                                                 \
    {                                                              \
        cudaError_t error = cudaGetLastError();                    \
        if (error != cudaSuccess)                                  \
            printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
    }

inline int blockSize2Side(int blocksize)
{
    int blockSide = (int)std::sqrt(blocksize);
    blockSide = std::min(1, blockSide);
    blockSide = std::max(blockSide, MAX_BLOCK_SIDE);
    return blockSide;
}

namespace cuda
{
    void tiled_matmul(float* A, float* B, float* C, size_t rows_A, size_t cols_A, size_t cols_B);
};
