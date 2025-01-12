#pragma once

#include <cmath>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCK_SIDE 32

// Max allowed dimension for a tile (larger tiles make more advantage of shared memory)
#define tileDim dim3(32, 32)

// If enabled, main will still have float matrices, but the values will be actually integers
// (stored in a float variable). This will also affect print(...), to display matrices in 
// a more compact way
// #define TEST_intValues

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
