#include "cuda.cuh"
#include "formatting.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "timer.h"

namespace cuda
{
    void matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
    {
        Info info = getInfo();

        float *dA, *dB, *dC;

        cudaMalloc(&dA, SIZE_A_BYTES);
        cudaMalloc(&dB, SIZE_B_BYTES);
        cudaMalloc(&dC, SIZE_C_BYTES);

        {
            ScopedTimer timer("A,B to device", POST);
            cudaMemcpy(dA, A, SIZE_A_BYTES, cudaMemcpyHostToDevice);
            cudaMemcpy(dB, B, SIZE_B_BYTES, cudaMemcpyHostToDevice);
        }

        {
            ScopedTimer timer("matmul kernel", POST);

            dim3 tiles(CEIL_DIV(COLS_C, tileDim.x), CEIL_DIV(ROWS_C, tileDim.y));

            dim3 gridDim = tiles;
            dim3 blockDim = tileDim;
            int sharedMemSize = 2 * dim2ToBytes(tileDim);

            // print info

            printf("gridDim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
            printf("blockDim: %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
            std::cout << "sharedMemSize per block: " << humanReadableMemSize(sharedMemSize) << " / " << humanReadableMemSize(info.sharedMemPerBlock) << std::endl;
            std::cout << "sharedMemSize per grid: " << humanReadableMemSize(sharedMemSize * gridDim.x * gridDim.y * gridDim.z) << " / " << humanReadableMemSize(info.sharedMemPerMultiprocessor * info.multiProcessorCount) << std::endl;

            kernels::tiled_matmul<<<gridDim, blockDim, sharedMemSize>>>(dA, dB, dC, rows_A, cols_A, cols_B);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        {
            ScopedTimer timer("C to host", POST);
            cudaMemcpy(C, dC, SIZE_C_BYTES, cudaMemcpyDeviceToHost);
        }

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}
