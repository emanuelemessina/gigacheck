#pragma once

#include "globals.h"
#include "mathdef.h"
#include <math.h>

#define tileDim dim3(globals::tileSide, globals::tileSide)

#ifndef TMP_CNT_FUNCTS
#define TMP_CNT_FUNCTS

#define getMatrixLinearIndex_cnt(mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col, flops, transfers) \
    {                                                                                                                 \
        (*flops) += 2;                                                                                                \
        if (tile_idy * tile_size + el_row >= mat_rows)                                                                \
        {                                                                                                             \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            (*flops) += 2;                                                                                            \
            if (tile_idx * tile_size + el_col >= mat_cols)                                                            \
            {                                                                                                         \
            }                                                                                                         \
            else                                                                                                      \
                                                                                                                      \
                (*flops) += 7;                                                                                        \
        }                                                                                                             \
    } // tot 11 op

#define getMatrixElement_cnt(matrix, mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col, flops, transfers) \
    {                                                                                                                     \
        getMatrixLinearIndex_cnt(mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col, flops, transfers);    \
        (*transfers)++;                                                                                                   \
    }

#define sumMatrixElement_cnt(matrix, mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col, val, flops, transfers) \
    {                                                                                                                          \
        getMatrixLinearIndex_cnt(mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col, flops, transfers);         \
        (*transfers)++;                                                                                                        \
        (*flops)++;                                                                                                            \
    }
#endif

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
        inline std::pair<uint64_t, uint64_t> compute_checksums_new(int N, float* matrix, int rows, int cols, ReductionDirection reduction_direction, float* checksum, dim3 blockDim)
        {
            uint64_t flops = 0, transfers = 0;

            // extern __shared__ float shared_data[]; // shared memory for intermediate sums, as big as the blockdim vector

            // int index_orthogonal = reduction_direction == ReductionDirection::ALONG_COL ? blockIdx.x : blockIdx.y; // 1 block per row/col checksum, the ortho direction is just the block index in that direction

            // float sum = 0.0f;

            // info along the reduction direction
            int index_reduction = 0;
            int blockDim_reduction = reduction_direction == ReductionDirection::ALONG_COL ? blockDim.y : blockDim.x;
            int limit_reduction = reduction_direction == ReductionDirection::ALONG_COL ? rows : cols;

            // // this thread accumulates values in blockdim offsets along the reduction direction
            for (int i = index_reduction; i < limit_reduction; i += blockDim_reduction, flops++) // 1 op * N/blockdim
            {
                transfers += 2;
                flops += 1 + (ReductionDirection::ALONG_COL ? 2 : 3);
                // sum += reduction_direction == ReductionDirection::ALONG_COL ? matrix[i * cols + index_orthogonal] : matrix[index_orthogonal * (cols + 1) + i]; // 1 transf, 3 ops | 4 ops (for each matmul, one of each type)
            }

            // // this thread stores his partial result into shared memory at his relative offset
            // shared_data[index_reduction] = sum; // 1 transf
            transfers += 1;
            // __syncthreads();                    // other threads do the same for their relative offset, we wait for all threads to finish

            // // this threads performs reduction over the shared memory vector
            for (int stride = blockDim_reduction / 2; stride > 0; stride /= 2) // 1 op one time, log2(blockdim) iterations
            {
                if (index_reduction < stride)
                    flops++;
                if (index_reduction < stride && index_reduction + stride < limit_reduction) // 1 op
                {
                    transfers += 2;
                    flops += 2;
                    // shared_data[index_reduction] += shared_data[index_reduction + stride]; // 2 ops, 2 transf
                }
                // __syncthreads(); // other threads do the same
            }

            // // finally, the first thread writes final reduction result to the matrix directly, or to the checksum array
            if (index_reduction == 0)
            {
                if (checksum != nullptr)
                {
                    // checksum[index_orthogonal] = shared_data[0]; // in the control checksum case: 2 transf
                    transfers += 1;
                } // in the load input checksums case

                else if (reduction_direction == ReductionDirection::ALONG_COL)
                {
                    // matrix[rows * cols + index_orthogonal] = shared_data[0]; // last row for column checksums // 2 transf, 2 ops
                    transfers += 1;
                    flops += 2;
                }
                else
                {
                    // matrix[index_orthogonal * (cols + 1) + cols] = shared_data[0]; // last column for row checksums // 2 transf, 3 ops
                    transfers += 1;
                    flops += 3;
                }
            }

            transfers *= N;
            flops *= N;

            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> compute_checksums(uint64_t N, uint64_t blockdim, ReductionDirection compute_direction, bool is_control)
        {
            bool col = compute_direction == ReductionDirection::ALONG_COL;
            bool row = !col;
            uint64_t flops = N * (N / blockdim) * 1 + N * (N / blockdim) * (N * 3 * col + N * 4 * row) + N * 1 + N * log2(blockdim) * (1 + 1 + 2) + 1 * (0 * is_control + 2 * col + 3 * row);
            uint64_t transfers = N * (N / blockdim) * 1 + N * 1 + N * log2(blockdim) * 2 + 2;

            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> find_checksum_mismatches(uint64_t N)
        {
            uint64_t flops = (N - 1) * 8 + 1 * 1;
            uint64_t transfers = (N - 1) * 3 + 1 * 2;
            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> find_checksum_mismatches_new(int N, const float* ec_matrix, int rows, int cols, float* control_checksum, ChecksumsToCompare checksums_to_compare, int* mismatch_count, int* mismatch_indexes, int* error_flag)
        {
            uint64_t flops = 0, transfers = 0;

            // int idx = blockIdx.x * blockDim.x + threadIdx.x;                                   // 2 op
            int idx = 0;
            flops += 2;

            int limit = checksums_to_compare == ChecksumsToCompare::ROW ? rows + 1 : cols + 1; // also verify checksums themselves  // 1 op
            flops++;

            if (idx >= limit || *error_flag)
            {
            } // 1 transf
            // return;

            // float mat_checksum_item = checksums_to_compare == ChecksumsToCompare::ROW ? ec_matrix[idx * (cols + 1) + cols] : ec_matrix[rows * (cols + 1) + idx]; // 1 transf, 3 ops
            flops += 3;
            transfers++;

            // float control_checksum_item = control_checksum[idx];                                                                                                 // 1 transfer
            transfers++;

            // float diff = abs(mat_checksum_item - control_checksum_item); // 1 op
            flops++;

#if CUDA_DEBUG_PRINT
            printf("%s thread %d: mat %f vs ctrl %f -> diff %f\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mat_checksum_item, control_checksum_item, diff);
#endif

            flops++; // product, then enter the if only once, as in ***
            // if (diff > max(0.0001 * mat_checksum_item, 1.0f)) // suppose 1 error per kernel, 1 thread in N triggers this // 1 op
            if (false)
            {
                // atomically update the mismatch count
                // int mismatch_idx = atomicAdd(mismatch_count, 1); // 1 transfer, 1 op
                // int mismatch_idx = 0;

                // SEE *

#if CUDA_DEBUG_PRINT
                printf("%s thread %d: current errors idx %d\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mismatch_idx);
#endif

                // Check if the mismatch array is full
                // if (mismatch_idx >= EDC_MAX_ERRORS) // suppose this is not triggered
                {
                    // mismatch array full: too many errors
                    // set error flag and stop further processing
                    *error_flag = 1; // 1 transfer
                    transfers++;

#if CUDA_DEBUG_PRINT
                    printf("%s thread %d: too many errors, raising error flag\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx + 1);
#endif

                    // return;
                }

                // mismatch_indexes[mismatch_idx] = idx; // 1 transfer
                transfers++;
            }

            flops *= N;
            transfers *= N;

            // ***
            transfers++;
            flops++;

            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> tiled_matmul(uint64_t N, uint64_t blockdim)
        {
            uint64_t flops = N * (N / blockdim) * ((1 + 2 + 2 + 11 + 2 + 2 + 11) + blockdim * 6) + N * 12;
            uint64_t transfers = N * (N / blockdim) * ((1 + 1 + 1 + 1) + blockdim * 2) + N * 1;
            return std::make_pair(flops, transfers);
        }

        inline std::pair<uint64_t, uint64_t> tiled_matmul_new(int N, float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, dim3 blockDim)
        {
            uint64_t flops = 0, transfers = 0;

            // extern __shared__ float shared_mem[];

            // float* shared_A = shared_mem;
            // float* shared_B = shared_mem + TILESIDE_BNK * TILESIDE; // 2 op
            flops += 3;

            float res = 0;

            // load tiles into shared memory

            for (int tile_idx = 0; tile_idx < CEIL_DIV(cols_A, blockDim.x); tile_idx++, flops++) // this thread looks through the tiles on his same row horizontally  // N/blockdim op
            {
                // load matrices elements into the shared tiles memory:
                //     - element (TH_ROW, TH_COL) of tile (TILE_IDY, tile_idx) of matrix A (one of A's tiles on the same grid row as this thread)
                //     - element (TH_ROW, TH_COL) of tile (tile_idx, TILE_IDX) of matrix B (the corresponding B tile, on the same grid column as the current row, and on the same grid row as the current column, because of how matrix multiplication works)
                // shared_A[TH_ROW * SKIPROWS_TILE + TH_COL] = getMatrixElement(A, rows_A, cols_A, TILESIDE, TILE_IDY, tile_idx, TH_ROW, TH_COL); // 2 op + 1 transf, 2 op + 11 op, 1 transf
                getMatrixElement_cnt(A, rows_A, cols_A, blockDim.x, 0, tile_idx, 0, 0, &flops, &transfers);
                flops += 2 + 2;
                transfers += 1;

                // shared_B[TH_ROW * SKIPROWS_TILE + TH_COL] = getMatrixElement(B, ROWS_B, cols_B, TILESIDE, tile_idx, TILE_IDX, TH_ROW, TH_COL); // 2 op + 1 transf, 2 op + 11 op, 1 transf
                getMatrixElement_cnt(B, cols_A, cols_B, blockDim.x, tile_idx, 0, 0, 0, &flops, &transfers);
                flops += 2 + 2;
                transfers += 1;

                // another thread on the same row as this one will look at the same tiles but load elements from his corresponding column in each tile chunk that he looks at
                // theads from other rows will do the same
                // we wait for all threads in the grid to finish loading their elements

                // __syncthreads();

                // at this point the tile for this block is loaded and the current thread can compute the scalar product for its tile

                // compute product on the shared memory tile

                for (int el = 0; el < blockDim.x; el++) // this thread looks through the elements of the tile horizontally   // blockdim op
                {
                    // res += shared_A[TH_ROW * SKIPROWS_TILE + el] * shared_B[TH_COL + el * SKIPROWS_TILE]; // tileA[TH_ROW, el] * tileB[el, TH_COL] , with the tile this thread belongs to  // 2 transf, 6 op
                    flops += 1 + (2 + 2) + 1 + (2 + 2);
                }

                // this thread will sum all the products in his tile into the result
                // at the next iteration, this thread will update the result with the product of the next tile

                // wait for all the threads to compute their scalar products

                // __syncthreads();

                // at the next iteration, this thread will load an element from an adjacent tile, all the thread combined will thus load the entire tile, and we can compute the scalar product for the next tile and add it to the result for this thread, and so on
            }

            // this thread will write his result into C, other threads will do the same for their positions
            sumMatrixElement_cnt(C, rows_A, cols_B, blockDim.x, 0, 0, 0, 0, res, &flops, &transfers); // 1 transf, 12 op

            flops *= N;
            transfers *= N;

            return std::make_pair(flops, transfers);
        }
    }
}
