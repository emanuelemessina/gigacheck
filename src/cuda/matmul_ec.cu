#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"
#include <math.h>

#define SWAP(a, b)    \
    {                 \
        auto tmp = a; \
        a = b;        \
        b = tmp;      \
    }

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
 * @param[in]   allocated_cols        The number of columns that are allocated for each matrix row (may be an overallocation of cols)
 * @param[in]   initial_offset        How much of the original matrix must be skipped at the beginning
 * @param[in]   next_row_offset       How much of the original matrix must be skipped to transition to the next row
 * @param[in]   leave_cell_after_row  If a cell should be left empty after each row (== copying B or a block of B)
 * @param[in]   stream                The stream on which to work
 */
void cp_matrix_to_CUDA(float* matrix, float* dst, int rows, int cols, int allocated_cols, int initial_offset, int next_row_offset, bool leave_cell_after_row, cudaStream_t stream)
{
    matrix += initial_offset;
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpyAsync(dst, matrix, cols * sizeof(float), cudaMemcpyHostToDevice, stream);

        matrix += next_row_offset;
        dst += allocated_cols + (leave_cell_after_row ? 1 : 0);
    }
}

/**
 * @brief The reverse of cp_matrix_to_CUDA
 *
 * @param[in]   matrix                The original, GPU matrix
 * @param[out]  dst                   The host memory allocated memory where to copy
 * @param[in]   rows                  The number of rows that should be copied
 * @param[in]   cols                  The number of columns that should be copied
 * @param[in]   allocated_cols        The number of columns that are allocated for each matrix row (may be an overallocation of cols)
 * @param[in]   initial_offset        How much of the host matrix must be skipped at the beginning
 * @param[in]   next_row_offset       How much of the host matrix must be skipped to transition to the next row
 * @param[in]   stream                The stream on which to work
 */
void cp_matrix_from_CUDA(float* matrix, float* dst, int rows, int cols, int allocated_cols, int initial_offset, int next_row_offset, cudaStream_t stream)
{
    dst += initial_offset;
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpyAsync(dst, matrix, cols * sizeof(float), cudaMemcpyDeviceToHost, stream);

        dst += next_row_offset;
        matrix += allocated_cols + 1;
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
 * @param[in]   strategy              Which strategy to use when matrices do not fit the GPU memory
 *
 */
void choose_division(int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim, Strategy strategy)
{
    int factA = 1, factB = 1, factC = 1;

    switch (strategy)
    {
        case bufferABC_forWriteback:
        case bufferABC_for2muls:
            factC = 2;

        case bufferAB:
            factA = factB = 2;
    }

    float required_mem = factA * (rows_A + 1) * cols_A;  // A with checksum
    required_mem += factB * ROWS_B * (cols_B + 1);       // B with checksum
    required_mem += factC * (ROWS_C + 1) * (COLS_C + 1); // C with checksum
    required_mem += factC * (COLS_C + 1);                // Column checksum buffer for C
    required_mem += factC * (ROWS_C + 1);                // Row checksum buffer for C
    required_mem *= sizeof(float);

    required_mem += 2 * EDC_MAX_ERRORS * sizeof(int); // Mismatches index buffer for error correction
    required_mem += 4 * sizeof(int);                  // Mismatch_count_x/y, error_x/y

    float exceeding_factor = required_mem / globals::maxGlobalMem;

    *num_split_common_dim = ceil(sqrt(exceeding_factor));
    *num_split_other_dim = ceil(exceeding_factor / (float)(*num_split_common_dim));
}

void copy_matrix_compute_checksum(float* h_mat, float* d_mat, int blockRow, int num_split_row, int blockCol, int num_split_col, int totRows, int totCols, int max_block_rows, int max_block_cols, cudaStream_t stream, char name)
{
    // copy to device
    int size = name == 'A' ? ((max_block_rows + 1) * max_block_cols * sizeof(float)) : (max_block_rows * (max_block_cols + 1) * sizeof(float));
    if (blockCol == num_split_col - 1 || blockRow == num_split_row - 1)
        cudaMemsetAsync(d_mat, 0, size, stream);

    int block_rows = CEIL_DIV(totRows, num_split_row);
    int block_cols = CEIL_DIV(totCols, num_split_col);
    int offset = blockRow * max_block_rows * totCols + blockCol * max_block_cols;

    if (blockCol == num_split_col - 1)
        block_cols = totCols - block_cols * blockCol;
    if (blockRow == num_split_row - 1)
        block_rows = totRows - block_rows * blockRow;

    cp_matrix_to_CUDA(h_mat, d_mat, block_rows, block_cols, max_block_cols, offset, totCols, name == 'B', stream);

    // calculate col checksums for A
    dim3 gridDim = name == 'A' ? dim3(max_block_cols) : dim3(1, max_block_rows);
    dim3 blockDim = name == 'A' ? dim3(1, tileDim.y) : dim3(tileDim.x, 1);
    int sharedMemSize = linearDimToBytes(name == 'A' ? tileDim.y : tileDim.x);
    kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream>>>(d_mat, max_block_rows, max_block_cols, name == 'A' ? ReductionDirection::ALONG_COL : ReductionDirection::ALONG_ROW);

    // Print mat (with checksums)
    if (globals::debugPrint)
        print_CUDA_matrix(
            d_mat,
            max_block_rows + (name == 'A' ? 1 : 0),
            max_block_cols + (name == 'A' ? 0 : 1),
            name == 'A' ? "A (w/ column checksum)" : "B (w/ column checksum)",
            name == 'A' ? HIGHLIGHT_LAST_ROW : HIGHLIGHT_LAST_COL,
            NULL,
            NULL,
            0);
}

namespace cuda
{
    EDCResult matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, int errors_count, int* error_xs, int* error_ys, float* error_values, Strategy strategy)
    {
        // How to split the matrices into blocks
        int num_split_common_dim;
        int num_split_other_dim;

        choose_division(rows_A, cols_A, cols_B, &num_split_common_dim, &num_split_other_dim, strategy);

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

        float *dA1, *dB1, *dC1, *dA2, *dB2, *dC2;

        // allocate checksums buffers
        float *d_cc_control1, *d_rc_control1, *d_cc_control2, *d_rc_control2;

        switch (strategy)
        {
            case bufferABC_forWriteback:
            case bufferABC_for2muls:
                cudaMalloc(&dC2, size_C_ec);

                cudaMalloc(&d_cc_control2, (COLS_C + 1) * sizeof(float));
                cudaMalloc(&d_rc_control2, (ROWS_C + 1) * sizeof(float));

            case bufferAB:
                cudaMalloc(&dA2, size_A_ec);
                cudaMalloc(&dB2, size_B_ec);

            case noBuffer:
                cudaMalloc(&dA1, size_A_ec);
                cudaMalloc(&dB1, size_B_ec);
                cudaMalloc(&dC1, size_C_ec);

                cudaMalloc(&d_cc_control1, (COLS_C + 1) * sizeof(float));
                cudaMalloc(&d_rc_control1, (ROWS_C + 1) * sizeof(float));
        }

        float* dA_cur = dA1;
        float* dA_alt = dA2;

        float* dB_cur = dB1;
        float* dB_alt = dB2;

        float* dC_cur = dC1;
        float* dC_alt = dC2;

        CUDA_CHECK

        // create streams for parallel executions
        cudaStream_t stream_A1;
        cudaStream_t stream_B1;
        cudaStream_t stream_C1;
        cudaStream_t stream_C1bis;
        cudaStream_t stream_A2;
        cudaStream_t stream_B2;
        cudaStream_t stream_C2;
        cudaStream_t stream_C2bis;

        cudaStream_t* stream_A_cur = &stream_A1;
        cudaStream_t* stream_A_alt = &stream_A2;

        cudaStream_t* stream_B_cur = &stream_B1;
        cudaStream_t* stream_B_alt = &stream_B2;

        cudaStream_t* stream_C_cur = &stream_C1;
        cudaStream_t* stream_C_alt = &stream_C2;

        cudaStream_t* stream_Cbis_cur = &stream_C1bis;
        cudaStream_t* stream_Cbis_alt = &stream_C2bis;

        switch (strategy)
        {
            case bufferABC_forWriteback:
            case bufferABC_for2muls:
                cudaStreamCreate(&stream_C2);
                cudaStreamCreate(&stream_C2bis);

            case bufferAB:
                cudaStreamCreate(&stream_A2);
                cudaStreamCreate(&stream_B2);

            case noBuffer:
                cudaStreamCreate(&stream_A1);
                cudaStreamCreate(&stream_B1);
                cudaStreamCreate(&stream_C1);
                cudaStreamCreate(&stream_C1bis);
        }

        // result
        bool result_correct = true;
        bool result_corrected = false;

        //
        int offset;

        int block_rows_C_cur;
        int block_cols_C_cur;

        int block_rows_C_alt;
        int block_cols_C_alt;

        // define threads organization
        dim3 gridDim;
        dim3 blockDim;
        int sharedMemSize;

        if (strategy != noBuffer)
        {
            copy_matrix_compute_checksum(A, dA_cur, 0, num_split_other_dim, 0, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_cur, 'A');
            copy_matrix_compute_checksum(B, dB_cur, 0, num_split_common_dim, 0, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_cur, 'B');
        }

        for (int C_row = 0; C_row < num_split_other_dim && result_correct; C_row++)
        {
            for (int C_col = 0; C_col < num_split_other_dim && result_correct; C_col++)
            {
                cudaMemsetAsync(dC_cur, 0, size_C_ec, *stream_C_cur);

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
                        if (strategy == noBuffer)
                        {
                            copy_matrix_compute_checksum(A, dA_cur, C_row, num_split_other_dim, block, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_cur, 'A');
                            copy_matrix_compute_checksum(B, dB_cur, block, num_split_common_dim, C_col, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_cur, 'B');
                        }

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // rows, cols for dC_cur
                    block_rows_C_cur = CEIL_DIV(ROWS_C, num_split_other_dim);
                    block_cols_C_cur = CEIL_DIV(COLS_C, num_split_other_dim);

                    if (C_row == num_split_other_dim - 1)
                        block_rows_C_cur = ROWS_C - block_rows_C_cur * C_row;
                    if (C_col == num_split_other_dim - 1)
                        block_cols_C_cur = COLS_C - block_cols_C_cur * C_col;

                    // compute the actual matrix multiplication as usual

                    {
                        ScopedTimer timer("tiled matmul", POST);

                        gridDim = dim3(CEIL_DIV(MAX_BLOCK_COLS_C + 1, tileDim.x), CEIL_DIV(MAX_BLOCK_ROWS_C + 1, tileDim.y));
                        blockDim = tileDim;
                        sharedMemSize = 2 * dim2ToBytes(tileDim);
                        kernels::tiled_matmul<<<gridDim, tileDim, sharedMemSize, *stream_C_cur>>>(dA_cur, dB_cur, dC_cur, max_block_rows_A + 1, max_block_cols_A, max_block_cols_B + 1);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // introduce errors in dC_cur
                    {
                        ScopedTimer timer("introduce error(s)", POST);

                        for (int i = 0; i < errors_count; i++)
                            cudaMemcpyAsync(dC_cur + error_ys[i] * (COLS_C + 1) + error_xs[i], &(error_values[i]), sizeof(float), cudaMemcpyHostToDevice, *stream_C_cur);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // print dC_cur (with mul checksums)
                    if (globals::debugPrint)
                        print_CUDA_matrix(dC_cur, MAX_BLOCK_ROWS_C + 1, MAX_BLOCK_COLS_C + 1, "C (w/ column checksum)", HIGHLIGHT_LAST_ROW_AND_COL, NULL, NULL, 0);

                    // compute control checksums after mul
                    {
                        ScopedTimer timer("compute control checksums", POST);

                        // compute col control checksum

                        gridDim = dim3(MAX_BLOCK_COLS_C + 1);
                        blockDim = dim3(1, tileDim.y);
                        sharedMemSize = linearDimToBytes(tileDim.y);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, *stream_C_cur>>>(dC_cur, MAX_BLOCK_ROWS_C, (MAX_BLOCK_COLS_C + 1), ReductionDirection::ALONG_COL, d_cc_control1);

                        // compute row control checksum

                        gridDim = dim3(1, MAX_BLOCK_ROWS_C + 1);
                        blockDim = dim3(tileDim.x, 1);
                        sharedMemSize = linearDimToBytes(tileDim.x);
                        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, *stream_Cbis_cur>>>(dC_cur, (MAX_BLOCK_ROWS_C + 1), MAX_BLOCK_COLS_C, ReductionDirection::ALONG_ROW, d_rc_control1);

                        cudaDeviceSynchronize();
                        CUDA_CHECK
                    }

                    // print control checksums
                    if (globals::debugPrint)
                    {
                        std::vector<int> zeros(errors_count, 0);
                        print_CUDA_matrix(d_rc_control1, MAX_BLOCK_ROWS_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL, zeros.data(), error_ys, errors_count);
                        print_CUDA_matrix(d_cc_control1, 1, MAX_BLOCK_COLS_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW, error_xs, zeros.data(), errors_count);
                    }

                    // edc

                    {
                        ScopedTimer timer("error detection (+ correction)", POST);

                        EDCResult edc_res = errors_detect_correct(dC_cur, MAX_BLOCK_ROWS_C, MAX_BLOCK_COLS_C, d_cc_control1, d_rc_control1, *stream_C_cur, *stream_Cbis_cur);

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

                    if (strategy != noBuffer)
                    {
                        // if strategy pre-loads A and B, and this is not the last iteration, pre-load the next A, B
                        if (block != (num_split_common_dim - 1) || C_row != (num_split_other_dim - 1) || C_col != (num_split_other_dim - 1))
                        {
                            int next_block = block + 1;
                            int next_C_col = C_col;
                            int next_C_row = C_row;
                            if (next_block == num_split_common_dim)
                            {
                                next_block = 0;
                                next_C_col = C_col + 1;
                                if (next_C_col == num_split_other_dim)
                                {
                                    next_C_col = 0;
                                    next_C_row = C_row + 1;
                                }
                            }
                            copy_matrix_compute_checksum(A, dA_alt, next_C_row, num_split_other_dim, next_block, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_alt, 'A');
                            copy_matrix_compute_checksum(B, dB_alt, next_block, num_split_common_dim, next_C_col, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_alt, 'B');
                        }
                    }

                    switch (strategy)
                    {
                        case bufferABC_forWriteback:

                        case bufferABC_for2muls:
                        case bufferAB:
                            SWAP(dA_cur, dA_alt)
                            SWAP(dB_cur, dB_alt)
                            SWAP(stream_A_cur, stream_A_alt)
                            SWAP(stream_B_cur, stream_B_alt)
                    }
                }
                // send back result (without checksums)

                {
                    ScopedTimer timer("C to host", POST);

                    offset = C_row * MAX_BLOCK_ROWS_C * COLS_C + C_col * MAX_BLOCK_COLS_C;

                    switch (strategy)
                    {
                        case bufferABC_forWriteback:
                            SWAP(dC_cur, dC_alt)
                            SWAP(stream_C_cur, stream_C_alt)
                            SWAP(stream_Cbis_cur, stream_Cbis_alt)
                            SWAP(block_rows_C_cur, block_rows_C_alt)
                            SWAP(block_cols_C_cur, block_cols_C_alt)

                            cp_matrix_from_CUDA(dC_alt, C, block_rows_C_alt, block_cols_C_alt, MAX_BLOCK_COLS_C, offset, COLS_C, *stream_C_alt);
                            break;

                        case bufferABC_for2muls:
                            break;

                        case bufferAB:
                        case noBuffer:
                            cp_matrix_from_CUDA(dC_cur, C, block_rows_C_cur, block_cols_C_cur, MAX_BLOCK_COLS_C, offset, COLS_C, *stream_C_cur);
                    }

                    // cudaDeviceSynchronize();

                    CUDA_CHECK
                }
            }
        }

        // cleanup:

        switch (strategy)
        {
            case bufferABC_forWriteback:
            case bufferABC_for2muls:
                cudaStreamDestroy(stream_C2);
                cudaStreamDestroy(stream_C2bis);

            case bufferAB:
                cudaStreamDestroy(stream_A2);
                cudaStreamDestroy(stream_B2);

            case noBuffer:
                cudaStreamDestroy(stream_A1);
                cudaStreamDestroy(stream_B1);
                cudaStreamDestroy(stream_C1);
                cudaStreamDestroy(stream_C1bis);
        }

        switch (strategy)
        {
            case bufferABC_forWriteback:
            case bufferABC_for2muls:
                cudaFree(dC2);

                cudaFree(d_cc_control2);
                cudaFree(d_rc_control2);

            case bufferAB:
                cudaFree(dA2);
                cudaFree(dB2);

            case noBuffer:
                cudaFree(dA1);
                cudaFree(dB1);
                cudaFree(dC1);

                cudaFree(d_cc_control1);
                cudaFree(d_rc_control1);
        }

        cudaHostUnregister(A);
        cudaHostUnregister(B);
        cudaHostUnregister(C);

        CUDA_CHECK

        if (!result_correct)
            return UNCORRECTABLE_ERROR;
        return result_corrected ? CORRECTED_ERROR : NO_ERROR;
    }
}
