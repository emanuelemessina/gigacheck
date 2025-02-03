#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"
#include <math.h>

/**
 * @brief Copies a matrix (or a portion of it) to CUDA memory
 *
 * It is able to handle also the copy of a block of a matrix.
 *
 * Assume we have to copy block (i, j) of a matrix HxW, divided into NxM blocks.
 *
 * The block (therefore also the GPU matrix) will have H/N rows and W/M cols.
 *
 * To copy the correct block, we have to start at an initial offset that includes:
 * - a delta of i * W * H/N, to select the correct row of blocks
 * - an extra delta of j * W/M to select the correct block within the row
 *
 * Moreover, if  M != 1 the next row of the matrix is not immediately after the previous one,
 * but it starts W cells after the previous one starts (next_row_offset)
 *
 * @param[in]   matrix                The original, host matrix
 * @param[out]  dst                   The GPU allocated memory where to copy
 * @param[in]   rows                  The number of rows that should be copied
 * @param[in]   cols                  The number of columns that should be copied
 * @param[in]   initial_offset        How much of the original matrix must be skipped at the beginning
 * @param[in]   next_row_offset       How much of the original matrix must be skipped to transition to the next row
 * @param[in]   leave_cell_after_row  If a cell should be left empty after each row (== copying B or a block of B)
 * @param[in]   stream                The stream on which to work
 */
void cp_matrix_to_CUDA(float* matrix, float* dst, int rows, int cols, int initial_offset, int next_row_offset, bool leave_cell_after_row, cudaStream_t stream)
{
    matrix += initial_offset;
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpyAsync(dst, matrix, cols * sizeof(float), cudaMemcpyHostToDevice, stream);

        matrix += next_row_offset;
        dst += cols + (leave_cell_after_row ? 1 : 0);
    }
}

/**
 * @brief The reverse of cp_matrix_to_CUDA
 *
 * @param[in]   matrix                The original, GPU matrix
 * @param[out]  dst                   The host memory allocated memory where to copy
 * @param[in]   rows                  The number of rows that should be copied
 * @param[in]   cols                  The number of columns that should be copied
 * @param[in]   initial_offset        How much of the host matrix must be skipped at the beginning
 * @param[in]   next_row_offset       How much of the host matrix must be skipped to transition to the next row
 * @param[in]   stream                The stream on which to work
 */
void cp_matrix_from_CUDA(float* matrix, float* dst, int rows, int cols, int initial_offset, int next_row_offset, cudaStream_t stream)
{
    dst += initial_offset;
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpyAsync(dst, matrix, cols * sizeof(float), cudaMemcpyDeviceToHost, stream);

        dst += next_row_offset;
        matrix += cols + 1;
    }
}

/**
 * @brief Given a matrix in global memory, prints it (by copying it to host memory at first)
 *
 * @param[in]  mat              The matrix to print
 * @param[in]  rows             Its number of rows
 * @param[in]  cols             Its number of columns
 * @param[in]  name             The name that should be printed
 * @param[in]  flags            Flags related to highlighting (as per matrix::print)
 * @param[in]  highlight_xs     Flags related to highlighting (as per matrix::print)
 * @param[in]  highlight_ys     Flags related to highlighting (as per matrix::print)
 * @param[in]  highlight_count  Flags related to highlighting (as per matrix::print)
 *
 */
void print_CUDA_matrix(float* mat, int rows, int cols, const char* name, int flags, int* highlight_xs, int* highlight_ys, int highlight_count)
{
    float* mat_host = matrix::alloc(rows, cols, false);
    cudaMemcpy(mat_host, mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    matrix::print(mat_host, rows, cols, name, flags, highlight_xs, highlight_ys, highlight_count);
    free(mat_host);
    CUDA_CHECK
}

/**
 * @brief Decideds if the matrices should be split in blocks to fit the global memory, and if so how many
 *
 * @param[in]   rows_A                #rows of matrix A
 * @param[in]   cols_A                #cols of matrix A
 * @param[in]   cols_B                #cols of matrix B
 * @param[out]  num_split_common_dim  The number of blocks for dividing A's columns and B's rows into (= the dimensions multiplied together)
 * @param[out]  num_split_other_dim   The number of blocks for dividing A's rows and B's columns into (= the "other" dimensions)
 *
 */
void choose_division(int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim)
{
    float required_mem = (rows_A + 1) * cols_A;  // A with checksum
    required_mem += ROWS_B * (cols_B + 1);       // B with checksum
    required_mem += (ROWS_C + 1) * (COLS_C + 1); // C with checksum
    required_mem += COLS_C + 1;                  // Column checksum buffer for C
    required_mem += ROWS_C + 1;                  // Row checksum buffer for C
    required_mem *= sizeof(float);

    required_mem += 2 * EDC_MAX_ERRORS * sizeof(int); // Mismatches index buffer for error correction
    required_mem += 4 * sizeof(int);                  // Mismatch_count_x/y, error_x/y

    float exceeding_factor = required_mem / globals::maxGlobalMem;

    *num_split_common_dim = ceil(sqrt(exceeding_factor));
    *num_split_other_dim = ceil(exceeding_factor / (float)(*num_split_common_dim));
}

namespace cuda
{
    EDCResult matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, int errors_count, int* error_xs, int* error_ys, float* error_values)
    {
        // How to split the matrices into blocks
        int num_split_common_dim;
        int num_split_other_dim;

        choose_division(rows_A, cols_A, cols_B, &num_split_common_dim, &num_split_other_dim);

        // Final sizes of matrices (excluding the checksums)
        int max_block_rows_A = CEIL_DIV(rows_A, num_split_other_dim);
        int max_block_cols_A = CEIL_DIV(cols_A, num_split_common_dim);
        int max_block_cols_B = CEIL_DIV(cols_B, num_split_other_dim);

        // register host pointers as pinned

        cudaHostRegister(A, SIZE_A_BYTES, cudaHostRegisterDefault);
        cudaHostRegister(B, SIZE_B_BYTES, cudaHostRegisterDefault);
        cudaHostRegister(C, SIZE_C_BYTES, cudaHostRegisterDefault);

        CUDA_CHECK

        // allocate device matrices with extra space for checksums A(m+1xn)B(nxp+1) = C(m+1xp+1)

        int size_A_ec = (max_block_rows_A + 1) * max_block_cols_A * sizeof(float);
        int size_B_ec = MAX_BLOCK_ROWS_B * (max_block_cols_B + 1) * sizeof(float);
        int size_C_ec = (MAX_BLOCK_ROWS_C + 1) * (MAX_BLOCK_COLS_C + 1) * sizeof(float);

        float *dA, *dB, *dC;

        cudaMalloc(&dA, size_A_ec);
        cudaMalloc(&dB, size_B_ec);
        cudaMalloc(&dC, size_C_ec);

        // allocate checksums buffers

        float *d_cc_control, *d_rc_control;

        cudaMalloc(&d_cc_control, (COLS_C + 1) * sizeof(float));
        cudaMalloc(&d_rc_control, (ROWS_C + 1) * sizeof(float));

        CUDA_CHECK

        // create streams for parallel executions
        cudaStream_t stream_A;
        cudaStream_t stream_B;
        cudaStream_t stream_C;
        cudaStream_t stream_Cbis;

        cudaStreamCreate(&stream_A);
        cudaStreamCreate(&stream_B);
        cudaStreamCreate(&stream_C);
        cudaStreamCreate(&stream_Cbis);

        // result
        bool result_correct = true;
        bool result_corrected = false;

        //
        int offset;

        int block_rows_A;
        int block_cols_A;

        int block_rows_B;
        int block_cols_B;

        int block_rows_C;
        int block_cols_C;

        // define threads organization
        dim3 gridDim;
        dim3 blockDim;
        int sharedMemSize;

        for (int C_row = 0; C_row < num_split_other_dim && result_correct; C_row++)
        {
            for (int C_col = 0; C_col < num_split_other_dim && result_correct; C_col++)
            {
                cudaMemsetAsync(dC, 0, size_C_ec, stream_C);

                for (int block = 0; block < num_split_common_dim && result_correct; block++)
                {

                    // send matrices to device and calculate checksums in parallel on different streams

                    // stream1: copyA, checkA
                    // stream2: copyB, checkB
                    // copies are sent to possibly the same copy queue, kernels to the same kernel queue (especially if only one queue per category exists)
                    // we use depth-first issue order: copyA - checkA - copyB - checkB
                    // breadth-first issue order would be: copyA - copyB - checkA - checkB
                    // we found that depth-first gives better performance

                    {
                        ScopedTimer timer("A,B to device + compute input checksums", POST);

                        // copy A to device
                        if (block == num_split_common_dim - 1 || C_row == num_split_other_dim - 1)
                            cudaMemsetAsync(dA, 0, size_A_ec, stream_A);

                        block_rows_A = CEIL_DIV(rows_A, num_split_other_dim);
                        block_cols_A = CEIL_DIV(cols_A, num_split_common_dim);
                        offset = C_row * max_block_rows_A * cols_A + block * max_block_cols_A;

                        if (block == num_split_common_dim - 1)
                            block_cols_A = cols_A - block_cols_A * block;
                        if (C_row == num_split_other_dim - 1)
                            block_rows_A = rows_A - block_rows_A * C_row;

                        cp_matrix_to_CUDA(A, dA, block_rows_A, block_cols_A, offset, cols_A, false, stream_A);

                        // calculate col checksums for A
                        gridDim = dim3(block_cols_A);
                        blockDim = dim3(1, tileDim.y);
                        sharedMemSize = linearDimToBytes(tileDim.y);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_A>>>(dA, block_rows_A, block_cols_A, ReductionDirection::ALONG_COL);

                        // Print dA (with checksums)
                        if (globals::debugPrint)
                            print_CUDA_matrix(dA, block_rows_A + 1, block_cols_A, "A (w/ column checksum)", HIGHLIGHT_LAST_ROW, NULL, NULL, 0);

                        // copy B to device
                        if (block == num_split_common_dim - 1 || C_col == num_split_other_dim - 1)
                            cudaMemsetAsync(dB, 0, size_B_ec, stream_B);

                        block_rows_B = CEIL_DIV(ROWS_B, num_split_common_dim);
                        block_cols_B = CEIL_DIV(cols_B, num_split_other_dim);
                        offset = block * MAX_BLOCK_ROWS_B * cols_B + C_col * max_block_cols_B;

                        if (block == num_split_common_dim - 1)
                            block_rows_B = ROWS_B - block_rows_B * block;
                        if (C_col == num_split_other_dim - 1)
                            block_cols_B = cols_B - block_cols_B * C_col;

                        cp_matrix_to_CUDA(B, dB, block_rows_B, block_cols_B, offset, cols_B, true, stream_B);

                        // calculate row checksums for B
                        gridDim = dim3(1, block_rows_B);
                        blockDim = dim3(tileDim.x, 1);
                        sharedMemSize = linearDimToBytes(tileDim.x);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_B>>>(dB, block_rows_B, block_cols_B, ReductionDirection::ALONG_ROW);

                        // print dB (with checksums)
                        if (globals::debugPrint)
                            print_CUDA_matrix(dB, block_rows_B, block_cols_B + 1, "B (w/ row checksum)", HIGHLIGHT_LAST_COL, NULL, NULL, 0);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // rows, cols for dC
                    block_rows_C = CEIL_DIV(ROWS_C, num_split_other_dim);
                    block_cols_C = CEIL_DIV(COLS_C, num_split_other_dim);

                    if (C_row == num_split_other_dim - 1)
                        block_rows_C = ROWS_C - block_rows_C * C_row;
                    if (C_col == num_split_other_dim - 1)
                        block_cols_C = COLS_C - block_cols_C * C_col;

                    // compute the actual matrix multiplication as usual

                    {
                        ScopedTimer timer("tiled matmul", POST);

                        gridDim = dim3(CEIL_DIV(block_cols_C + 1, tileDim.x), CEIL_DIV(block_rows_C + 1, tileDim.y));
                        blockDim = tileDim;
                        sharedMemSize = 2 * dim2ToBytes(tileDim);
                        kernels::tiled_matmul<<<gridDim, tileDim, sharedMemSize, stream_C>>>(dA, dB, dC, block_rows_A + 1, block_cols_A, block_cols_B + 1);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // introduce errors in dC
                    {
                        ScopedTimer timer("introduce error(s)", POST);

                        for (int i = 0; i < errors_count; i++)
                            cudaMemcpyAsync(dC + error_ys[i] * (COLS_C + 1) + error_xs[i], &(error_values[i]), sizeof(float), cudaMemcpyHostToDevice, stream_C);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // print dC (with mul checksums)
                    if (globals::debugPrint)
                        print_CUDA_matrix(dC, block_rows_C + 1, block_cols_C + 1, "C (w/ column checksum)", HIGHLIGHT_LAST_ROW_AND_COL, NULL, NULL, 0);

                    // compute control checksums after mul
                    {
                        ScopedTimer timer("compute control checksums", POST);

                        // compute col control checksum

                        gridDim = dim3(block_cols_C + 1);
                        blockDim = dim3(1, tileDim.y);
                        sharedMemSize = linearDimToBytes(tileDim.y);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_C>>>(dC, block_rows_C, (block_cols_C + 1), ReductionDirection::ALONG_COL, d_cc_control);

                        // compute row control checksum

                        gridDim = dim3(1, block_rows_C + 1);
                        blockDim = dim3(tileDim.x, 1);
                        sharedMemSize = linearDimToBytes(tileDim.x);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_Cbis>>>(dC, (block_rows_C + 1), block_cols_C, ReductionDirection::ALONG_ROW, d_rc_control);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // print control checksums
                    if (globals::debugPrint)
                    {
                        std::vector<int> zeros(errors_count, 0);
                        print_CUDA_matrix(d_rc_control, block_rows_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL, zeros.data(), error_ys, errors_count);
                        print_CUDA_matrix(d_cc_control, 1, block_cols_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW, error_xs, zeros.data(), errors_count);
                    }

                    // edc

                    {
                        ScopedTimer timer("error detection (+ correction)", POST);

                        EDCResult edc_res = errors_detect_correct(dC, block_rows_C, block_cols_C, d_cc_control, d_rc_control, stream_C, stream_Cbis);

                        // choice: don't send back the result if it's wrong
                        // NOTE: now the result may be partial, since an error will stop the rest
                        switch (edc_res)
                        {
                            case UNCORRECTABLE_ERROR:
                                result_correct = false;
                                break;

                            case CORRECTED_ERROR:
                                result_corrected = true;
                                break;

                            case NO_ERROR:
                                break;
                        }
                    }

                    // send back result (without checksums)

                    {
                        ScopedTimer timer("C to host", POST);

                        offset = C_row * MAX_BLOCK_ROWS_C * COLS_C + C_col * MAX_BLOCK_COLS_C;

                        cp_matrix_from_CUDA(dC, C, block_rows_C, block_cols_C, offset, COLS_C, stream_C);

                        // for (int r = 0; r < ROWS_C; ++r) // copy row without last column
                        //     cudaMemcpyAsync(C + r * COLS_C, dC + r * (COLS_C + 1), COLS_C * sizeof(float), cudaMemcpyDeviceToHost, stream_C);

                        cudaDeviceSynchronize();

                        CUDA_CHECK
                    }
                }
            }
        }

        // cleanup:

        // cudaStreamCreate(&stream_A);
        // cudaStreamCreate(&stream_B);

        // cudaStreamSynchronize(stream_A);
        // cudaStreamSynchronize(stream_B);
        // cudaStreamSynchronize(stream_B);
        // cudaStreamSynchronize(stream_B);

        cudaStreamDestroy(stream_A);
        cudaStreamDestroy(stream_B);
        cudaStreamDestroy(stream_C);
        cudaStreamDestroy(stream_Cbis);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaHostUnregister(A);
        cudaHostUnregister(B);
        cudaHostUnregister(C);

        CUDA_CHECK

        if (!result_correct)
            return UNCORRECTABLE_ERROR;
        return result_corrected ? CORRECTED_ERROR : NO_ERROR;
    }
}
