#pragma once

#include <string>

namespace cuda
{
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
     * @brief Computes the product between two matrices A and B, exploiting the GPU
     * with the tiled multiplication algorithm
     *
     * @param[in]   A       The first matrix to mutiply
     * @param[in]   B       The second matrix to multiply
     * @param[out]  C       The result matrix
     * @param[in]   rows_A  #rows in A
     * @param[in]   cols_A  #cols in A
     * @param[in]   cols_B  #cols in B
     *
     */
    void matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);

    /**
     * @brief Computes the error corrected tiled matrix multiplication between A and B
     *
     * @param[in]   A       First operand matrix
     * @param[in]   B       Second operand matrix
     * @param[out]  C       Result matrix
     * @param[in]   rows_A  #rows in A
     * @param[in]   cols_A  #cols in A
     * @param[in]   cols_B  #cols in B
     * @param[in]   print_intermediate  whether to print the matrix with checksums
     *
     */
    void matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);
};
