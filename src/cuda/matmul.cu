#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "parameters.h"

namespace cuda
{
    void matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
    {
        float *dA, *dB, *dC;

        cudaMalloc(&dA, SIZE_A_BYTES);
        cudaMalloc(&dB, SIZE_B_BYTES);
        cudaMalloc(&dC, SIZE_C_BYTES);

        cudaMemcpy(dA, A, SIZE_A_BYTES, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, SIZE_B_BYTES, cudaMemcpyHostToDevice);

        dim3 tiles(CEIL_DIV(COLS_C, tileDim.x), CEIL_DIV(ROWS_C, tileDim.y));

        dim3 gridSize = tiles;
        dim3 blockSize = tileDim;
        int sharedMemSize = 2 * dim3ToBytes(blockSize);

        kernels::tiled_matmul<<<gridSize, blockSize, sharedMemSize>>>(dA, dB, dC, rows_A, cols_A, cols_B);
        cudaDeviceSynchronize();
        CUDA_CHECK

        cudaMemcpy(C, dC, SIZE_C_BYTES, cudaMemcpyDeviceToHost);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}