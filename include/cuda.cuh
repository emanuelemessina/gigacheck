#pragma once

#include "edc.cuh"
#include "iomod.h"
#include <stdio.h>
#include <string>

#define CUDA_CHECK                                                                       \
    {                                                                                    \
        cudaError_t error = cudaGetLastError();                                          \
        if (error != cudaSuccess)                                                        \
            CERR << RED << "CUDA Error: " << cudaGetErrorString(error) << RESET << ENDL; \
    }

#define CUDA_CREATE_RECORD_EVENT(event, stream) \
    {                                           \
        cudaEventCreate(&event);                \
        cudaEventRecord(event, stream);         \
    }

#define CUDA_WAIT_EVENT_DESTROY(event, stream) \
    {                                          \
        cudaStreamWaitEvent(stream, event);    \
        cudaEventDestroy(event);               \
    }

#define CUDA_WAIT_EVENT_DESTROY_IF(event, stream, destroy) \
    {                                                      \
        cudaStreamWaitEvent(stream, event);                \
        if (destroy)                                       \
            cudaEventDestroy(event);                       \
    }

namespace cuda
{
    typedef enum
    {
        simple,
        preloadAB,
        preloadAB_deferUnloadC,
        parallelMul
    } MulStrategy;

    struct Info
    {
        int deviceNumber;
        std::string deviceName;
        int multiProcessorCount;
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        size_t sharedMemPerMultiprocessor;
    };

    /**
     * @brief Get information about the GPU
     * @return CUDAInfo struct with information about the GPU
     */
    Info getInfo();

    /**
     * @brief Computes the error corrected tiled matrix multiplication between A and B
     *
     * @param[in]   A             First operand matrix
     * @param[in]   B             Second operand matrix
     * @param[out]  C             Result matrix
     * @param[in]   rows_A        #rows in A
     * @param[in]   cols_A        #cols in A
     * @param[in]   cols_B        #cols in B
     * @param[in]   errors_count  Number of errors to introduce
     * @param[in]   error_xs      buffer with x coords of the errors
     * @param[in]   error_ys      buffer Y coords of the errors
     * @param[in]   error_values  buffer with the values of the errors
     * @param[in]   strategy      which strategy to use when matrices do not fit the GPU memory
     */
    EDCResult matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, int errors_count, int** error_xs, int** error_ys, float** error_values, MulStrategy strategy);
};
