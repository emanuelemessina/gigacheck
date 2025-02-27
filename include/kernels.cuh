#pragma once

#include "globals.h"
#include "mathdef.h"
#include <math.h>

#define tileDim dim3(globals::tileSide, globals::tileSide)

/**
 * @brief Calculates a bank-conflict-free size in bytes for float shared memory from a dimension along an axis.
 *
 * @param[in]  dim  eg. 32 would output (32 + 1) * sizeof(float)
 */
inline int linearDimToBytes(int dim)
{
    return (dim + (dim % 2 == 0 ? 1 : 0)) * sizeof(float);
}

/**
 * @brief Calculates a bank-conflict-free size in bytes for float shared memory from a 2d tileDim (default blocksize).
 *
 * @param[in]  dim2
 */
inline int dim2ToBytes(dim3 dim2)
{
    return linearDimToBytes(dim2.x) * dim2.y;
}

inline uint64_t dimsToN(dim3 gridDim, dim3 blockDim)
{
    return gridDim.x * gridDim.y * gridDim.z * blockDim.x + blockDim.y * blockDim.z;
}

enum ReductionDirection
{
    ALONG_COL,
    ALONG_ROW
};

enum ChecksumsToCompare
{
    COL = ReductionDirection::ALONG_COL,
    ROW = ReductionDirection::ALONG_ROW
};

namespace kernels
{
    /**
     * @brief Computes checksums for a given matrix.
     *
     * This CUDA kernel computes checksums for a given matrix. The checksums can be computed
     * either for rows or columns based on the `reduction_direction` parameter.
     *
     * @note The checksum is stored, unless the checksum vector is provided, in the last row (for column checksums) or the last column (for row checksums) directly in the matrix itself. Thus the matrix is assumed to have 1 extra row or column than what specified by the `rows` and `cols` parameters.
     *
     * @param matrix Pointer to the matrix data stored in device memory.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param checksum If provided, the checksum vector is stored here instead of in the matrix.
     * @param compute_direction Flag indicating the mode of checksum computation.
     */
    __global__ void compute_checksums(float* matrix, int rows, int cols, ReductionDirection compute_direction, float* checksum = nullptr);

    /**
     * @brief CUDA kernel to compute the product between two matrices, using the tiled shared
     * memory approach
     *
     * @param[in]   A       The first matrix to multiply
     * @param[in]   B       The second matrix to multiply
     * @param[out]  C       The matrix where to write the result
     * @param[in]   rows_A  #rows in A
     * @param[in]   cols_A  #cols in A
     * @param[in]   cols_B  #cols in B
     */
    __global__ void tiled_matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B);

    /**
     * @brief Kernel function to find checksum mismatches in an error correction matrix.
     *
     * This kernel compares the row or col checksums in the error correction matrix with a given control checksum
     * and identifies mismatches. It updates the mismatch count and records the indexes of mismatched rows/columns.
     *
     * @param ec_matrix Pointer to the error correction matrix.
     * @param rows Number of rows in the matrix (checksum excluded).
     * @param cols Number of columns in the matrix (checksum excluded).
     * @param control_checksum Pointer to the control checksum.
     * @param checksums_to_compare Which checksum to compare (row or column).
     * @param[out] mismatch_count Will contain the number of mismatches encountered.
     * @param[out] mismatch_indexes Will contain the indexes of the mismatches.
     * @param[out] error_flag Will be set to 1 if there is an internal kernel error (eg. encountered more mismatches than allowed).
     */
    __global__ void find_checksum_mismatches(const float* ec_matrix, int rows, int cols, float* control_checksum, ChecksumsToCompare checksums_to_compare, int* mismatch_count, int* mismatch_indexes, int* error_flag);

    /**
     * @brief This namespace contains a homonimous function for each kernel, that returns the number of flops and transfers for that kernel as a pair.
     *
     */
    namespace metrics
    {
        inline std::pair<uint64_t, uint64_t> compute_checksums(uint64_t N, uint64_t blockdim, int rows, int cols, ReductionDirection dir)
        {
            uint64_t flops = N + (dir == ReductionDirection::ALONG_COL ? cols : rows) * (blockdim - 1);
            uint64_t transfers = N + (dir == ReductionDirection::ALONG_COL ? cols : rows) * 1;
            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> find_checksum_mismatches(uint64_t N)
        {
            uint64_t flops = N * (2 + 2);
            uint64_t transfers = N * 2 + 1;
            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> tiled_matmul(uint64_t N, uint64_t blockdim, int cols_A)
        {
            uint64_t flops = N * (cols_A * 2 + 1);
            uint64_t transfers = N * (cols_A / blockdim) * 2 + N * 1;
            return std::make_pair(flops, transfers);
        }
    }
}
