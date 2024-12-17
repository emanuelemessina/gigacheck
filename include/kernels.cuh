#pragma once

#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCK_SIDE 32

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
    float** strassen_parallelizing_recursion(float** A, float** B, int rowsA, int colsA, int colsB);
};
