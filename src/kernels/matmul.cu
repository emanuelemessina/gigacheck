#include "kernels.cuh"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(float* A, float* B, float* C, size_t N)
{
    /*
    __shared__ float shared_A[blockDim.y][blockDim.x + 1];
    __shared__ float shared_B[blockDim.y][blockDim.x + 1];
    */

    extern __shared__ float shared_mem[]; // Dynamically allocated shared memory

    // Split the shared memory into two 2D arrays for A and B
    float* shared_A = shared_mem;
    float* shared_B = shared_mem + blockDim.y * (blockDim.x + 1); // +1 to avoid bank conflict

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N)
        return;

    float result = 0.0f;

    // each thread loads its relative position for each matrix tile into the local block tile

    /*
        o = this thread, once per tile iteration

        A:
        ----------------
        |....|....|....|
        |..o.|..o.|..o.|
        |....|....|....|
        ----------------
        B:
        ----------------
        |..o.|....|....|
        |..o.|....|....|
        |..o.|....|....|
        ----------------

        @ tile iteration

        u,i,o,p = threads in block

        shared tiles:
        |uiop|  --
                u
                --
                i
                --
                o
                --
                p
                --

        result += sharedA dot sharedB

        sum for all the tile positions

        after accumulating the partial dots for all the tiles

        C:
        ----------------
        |...     .|....|
        |..result.|....|
        |...     .|....|
        ----------------
        |...     .|....|
        |..      .|....|
        |...     .|....|
        ----------------

        for example thread u in the same iteration performs the dot between the same row as o on A but another column of B
        each thread calculates the partial dot product for a tile and accumulates the result in its global matrix position
    */

    for (int tileIdx = 0; tileIdx < N / blockDim.x; ++tileIdx)
    {
        int rowOffset = tileIdx * blockDim.x + threadIdx.x;
        int colOffset = tileIdx * blockDim.y + threadIdx.y;

        int sharedIdx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;

        if (rowOffset < N)
            shared_A[sharedIdx] = A[row * N + rowOffset];
        else
            shared_A[sharedIdx] = 0.0f; // important to fill padding with 0 because the sum will occur over all the tile spots

        if (colOffset < N)
            shared_B[sharedIdx] = B[colOffset * N + col];
        else
            shared_B[sharedIdx] = 0.0f;

        // all threads have loaded their tile row/col
        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k)
        {
            result += shared_A[threadIdx.y * (blockDim.x + 1) + k] * shared_B[k * (blockDim.x + 1) + threadIdx.x];
        }

        // wait for all because a this thread can disrupt the tile while another is performing the dot
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = result;
    }
}

namespace cuda
{

    void matmul(const float* ha, const float* hb, float* hc, int N, int blocksize)
    {

        float *da, *db, *dc;

        // Allocate memory on the device (GPU)
        size_t bytes = N * N * sizeof(float);
        cudaMalloc(&da, bytes);
        cudaMalloc(&db, bytes);
        cudaMalloc(&dc, bytes);

        {
            ScopedTimer t1("memcpy inputs CPU -> GPU", POST);

            // Copy data to device
            cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
        }

        // Define block and grid size
        int blockSide = blockSize2Side(blocksize);
        dim3 blockSize(blockSide, blockSide);
        int gridSide = (N + blockSide - 1) / blockSide;
        dim3 gridSize(gridSide, gridSide);

        {
            size_t sharedMemSize = (blockSide + 1) * blockSide * 2 * sizeof(float);
            ScopedTimer t2("kernel execution", POST);
            kernel<<<gridSize, blockSize, sharedMemSize>>>(da, db, dc, N);
            CUDA_CHECK
            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();
        }

        {
            ScopedTimer t3("memcpy output GPU -> CPU", POST);

            // Copy result vector c from device to host
            cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
        }

        // Free the allocated memory on the device
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }

}
