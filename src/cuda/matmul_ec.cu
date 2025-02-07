#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"

#define SWAP(a, b)    \
    {                 \
        auto tmp = a; \
        a = b;        \
        b = tmp;      \
    }

#define CUDA_WAIT_DESTROY_EVENT(event, stream, destroy) \
    {                                                   \
        cudaStreamWaitEvent(stream, event);             \
        if (destroy)                                    \
            cudaEventDestroy(event);                    \
    }

#define CUDA_CREATE_RECORD_EVENT(event, stream) \
    {                                           \
        cudaEventCreate(&event);                \
        cudaEventRecord(event, stream);         \
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
        matrix += allocated_cols;
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

void copy_matrix_compute_checksum(float* h_mat, float* d_mat, int blockRow, int num_split_row, int blockCol, int num_split_col, int totRows, int totCols, int max_block_rows, int max_block_cols, cudaStream_t stream, char name, bool without_checksum)
{
    // copy to device
    int extra = without_checksum ? 0 : 1;
    int size = name == 'A' ? ((max_block_rows + extra) * max_block_cols * sizeof(float)) : (max_block_rows * (max_block_cols + extra) * sizeof(float));
    if (blockCol == num_split_col - 1 || blockRow == num_split_row - 1)
        cudaMemsetAsync(d_mat, 0, size, stream);

    int block_rows = CEIL_DIV(totRows, num_split_row);
    int block_cols = CEIL_DIV(totCols, num_split_col);
    int offset = blockRow * max_block_rows * totCols + blockCol * max_block_cols;

    if (blockCol == num_split_col - 1)
        block_cols = totCols - block_cols * blockCol;
    if (blockRow == num_split_row - 1)
        block_rows = totRows - block_rows * blockRow;

    cp_matrix_to_CUDA(h_mat, d_mat, block_rows, block_cols, max_block_cols, offset, totCols, name == 'B' && !without_checksum, stream);

    if (!without_checksum)
    {
        // calculate col checksums for A
        dim3 gridDim = name == 'A' ? dim3(max_block_cols) : dim3(1, max_block_rows);
        dim3 blockDim = name == 'A' ? dim3(1, tileDim.y) : dim3(tileDim.x, 1);
        int sharedMemSize = linearDimToBytes(name == 'A' ? tileDim.y : tileDim.x);
        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream>>>(d_mat, max_block_rows, max_block_cols, name == 'A' ? ReductionDirection::ALONG_COL : ReductionDirection::ALONG_ROW);
    }

    // Print mat (with checksums)
    if (globals::debugPrint)
        print_CUDA_matrix(
            d_mat,
            max_block_rows + (name == 'A' ? extra : 0),
            max_block_cols + (name == 'A' ? 0 : extra),
            without_checksum ? (name == 'A' ? "A" : "B") : (name == 'A' ? "A (w/ column checksum)" : "B (w/ column checksum)"),
            without_checksum ? 0 : (name == 'A' ? HIGHLIGHT_LAST_ROW : HIGHLIGHT_LAST_COL),
            NULL,
            NULL,
            0);
}

void C_compute_checksum(float* C, ReductionDirection direction, int max_block_cols_B, int max_block_rows_A, cudaStream_t stream, float* result_array)
{
    // compute col control checksum

    if (direction == ReductionDirection::ALONG_COL)
    {
        dim3 gridDim = dim3(MAX_BLOCK_COLS_C + 1);
        dim3 blockDim = dim3(1, tileDim.y);
        int sharedMemSize = linearDimToBytes(tileDim.y);
        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream>>>(C, MAX_BLOCK_ROWS_C, (MAX_BLOCK_COLS_C + 1), ReductionDirection::ALONG_COL, result_array);
    }

    // compute row control checksum
    else
    {
        dim3 gridDim = dim3(1, MAX_BLOCK_ROWS_C + 1);
        dim3 blockDim = dim3(tileDim.x, 1);
        int sharedMemSize = linearDimToBytes(tileDim.x);
        kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream>>>(C, (MAX_BLOCK_ROWS_C + 1), MAX_BLOCK_COLS_C, ReductionDirection::ALONG_ROW, result_array);
    }
}

namespace cuda
{
    void C_mult_check_correct(float* A, float* B, float* C, int rows_A, int cols_B, int* block_rows_C_cur, int* block_cols_C_cur, int C_row, int C_col, int block, int max_block_rows_A, int max_block_cols_A, int max_block_cols_B, cudaStream_t stream, cudaStream_t streamBis, int num_split_common_dim, int num_split_other_dim, int errors_count, int* error_xs, int* error_ys, float* error_values, bool* result_correct, bool* result_corrected, bool without_error_check)
    {
        int extra = without_error_check ? 0 : 1;

        // rows, cols for dC_cur
        (*block_rows_C_cur) = CEIL_DIV(ROWS_C, num_split_other_dim);
        (*block_cols_C_cur) = CEIL_DIV(COLS_C, num_split_other_dim);

        if (C_row == num_split_other_dim - 1)
            (*block_rows_C_cur) = ROWS_C - (*block_rows_C_cur) * C_row;
        if (C_col == num_split_other_dim - 1)
            (*block_cols_C_cur) = COLS_C - (*block_cols_C_cur) * C_col;

        // compute the actual matrix multiplication as usual

        {
            ScopedTimer timer("tiled matmul", POST);

            dim3 gridDim = dim3(CEIL_DIV(MAX_BLOCK_COLS_C + extra, tileDim.x), CEIL_DIV(MAX_BLOCK_ROWS_C + extra, tileDim.y));
            int sharedMemSize = 2 * dim2ToBytes(tileDim);
            kernels::tiled_matmul<<<gridDim, tileDim, sharedMemSize, stream>>>(A, B, C, max_block_rows_A + extra, max_block_cols_A, max_block_cols_B + extra);

            CUDA_CHECK
        }

        if (without_error_check)
        {
            if (globals::debugPrint)
                print_CUDA_matrix(C, MAX_BLOCK_ROWS_C + extra, MAX_BLOCK_COLS_C, "C", 0, NULL, NULL, 0);
            return;
        }

        float *d_cc_control, *d_rc_control;
        cudaMalloc(&d_cc_control, (MAX_BLOCK_COLS_C + 1) * sizeof(float));
        cudaMalloc(&d_rc_control, (MAX_BLOCK_ROWS_C + 1) * sizeof(float));

        cudaEvent_t C_err_added;
        cudaEventCreate(&C_err_added);

        // introduce errors in dC_cur
        {
            ScopedTimer timer("introduce error(s)", POST);

            for (int i = 0; i < errors_count; i++)
            {
                float tmp;
                cudaMemcpy(&tmp, C + error_ys[i] * (MAX_BLOCK_COLS_C + 1) + error_xs[i], sizeof(float), cudaMemcpyDeviceToHost);
                tmp += error_values[i];
                cudaMemcpy(C + error_ys[i] * (MAX_BLOCK_COLS_C + 1) + error_xs[i], &tmp, sizeof(float), cudaMemcpyHostToDevice);
            }
            cudaEventRecord(C_err_added, stream);

            CUDA_CHECK
        }

        // print dC_cur (with mul checksums)
        if (globals::debugPrint)
            print_CUDA_matrix(C, MAX_BLOCK_ROWS_C + 1, MAX_BLOCK_COLS_C + 1, "C (w/ column checksum)", HIGHLIGHT_LAST_ROW_AND_COL, error_xs, error_ys, errors_count);

        // compute control checksums after mul
        {
            ScopedTimer timer("compute control checksums", POST);

            // compute col control checksum
            C_compute_checksum(C, ReductionDirection::ALONG_COL, max_block_cols_B, max_block_rows_A, stream, d_cc_control);

            // compute row control checksum
            cudaStreamWaitEvent(streamBis, C_err_added);
            C_compute_checksum(C, ReductionDirection::ALONG_ROW, max_block_cols_B, max_block_rows_A, streamBis, d_rc_control);

            CUDA_CHECK
        }

        // print control checksums
        if (globals::debugPrint)
        {
            std::vector<int> zeros(errors_count, 0);
            print_CUDA_matrix(d_rc_control, MAX_BLOCK_ROWS_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL, zeros.data(), error_ys, errors_count);
            print_CUDA_matrix(d_cc_control, 1, MAX_BLOCK_COLS_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW, error_xs, zeros.data(), errors_count);
        }

        // edc

        {
            ScopedTimer timer("error detection (+ correction)", POST);

            bool recompute_vertical_checksums = false;
            bool recompute_horizontal_checksums = false;

            EDCResult edc_res = errors_detect_correct(C, MAX_BLOCK_ROWS_C, MAX_BLOCK_COLS_C, d_cc_control, d_rc_control, stream, streamBis, &recompute_vertical_checksums, &recompute_horizontal_checksums);

            // choice: don't send back the result if it's wrong
            // NOTE: now the result may be partial, since an error will stop the rest
            switch (edc_res)
            {
                case UNCORRECTABLE_ERROR:
                    *result_correct = false;
                    break;

                case CORRECTED_ERROR:
                    *result_corrected = true;
                    break;

                case NO_ERROR:
                    break;
            }

            if (recompute_horizontal_checksums)
                C_compute_checksum(C, ReductionDirection::ALONG_ROW, max_block_cols_B, max_block_rows_A, stream, NULL);

            if (recompute_vertical_checksums)
                C_compute_checksum(C, ReductionDirection::ALONG_COL, max_block_cols_B, max_block_rows_A, stream, NULL);
        }

        cudaFree(d_cc_control);
        cudaFree(d_rc_control);
        cudaEventDestroy(C_err_added);
    }

    EDCResult matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, int errors_count, int** error_xs, int** error_ys, float** error_values, Strategy strategy, bool without_error_check)
    {
        // How to split the matrices into blocks
        int num_split_common_dim;
        int num_split_other_dim;

        matrix::choose_division(rows_A, cols_A, cols_B, &num_split_common_dim, &num_split_other_dim, strategy);

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

        int extra = without_error_check ? 0 : 1;
        int size_A_ec = (max_block_rows_A + extra) * max_block_cols_A * sizeof(float);
        int size_B_ec = MAX_BLOCK_ROWS_B * (max_block_cols_B + extra) * sizeof(float);
        int size_C_ec = (MAX_BLOCK_ROWS_C + extra) * (MAX_BLOCK_COLS_C + extra) * sizeof(float);

        float *dA1, *dB1, *dC1, *dA2, *dB2, *dC2;

        switch (strategy)
        {
            case preloadAB_deferUnloadC:
            case parallelMul:
                cudaMalloc(&dC2, size_C_ec);

            case preloadAB:
                if (strategy != parallelMul)
                    cudaMalloc(&dA2, size_A_ec);
                cudaMalloc(&dB2, size_B_ec);

            case simple:
                cudaMalloc(&dA1, size_A_ec);
                cudaMalloc(&dB1, size_B_ec);
                cudaMalloc(&dC1, size_C_ec);
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
            case preloadAB_deferUnloadC:
            case parallelMul:
                cudaStreamCreate(&stream_C2);
                cudaStreamCreate(&stream_C2bis);

            case preloadAB:
                cudaStreamCreate(&stream_A2);
                cudaStreamCreate(&stream_B2);

            case simple:
                cudaStreamCreate(&stream_A1);
                cudaStreamCreate(&stream_B1);
                cudaStreamCreate(&stream_C1);
                cudaStreamCreate(&stream_C1bis);
        }

        // result
        bool result_correct = true;
        bool result_corrected = false;
        bool result_correct_alt = true;
        bool result_corrected_alt = false;

        //
        int offset;

        int block_rows_C_cur;
        int block_cols_C_cur;

        int block_rows_C_alt;
        int block_cols_C_alt;

        // Sync events
        cudaEvent_t A_copied;
        cudaEvent_t B_copied;
        cudaEvent_t A_alt_copied;
        cudaEvent_t B_alt_copied;

        cudaEvent_t A_can_be_overwritten;
        cudaEvent_t B_can_be_overwritten;
        cudaEvent_t B_alt_can_be_overwritten;

        CUDA_CREATE_RECORD_EVENT(A_can_be_overwritten, *stream_A_cur);
        CUDA_CREATE_RECORD_EVENT(B_can_be_overwritten, *stream_B_cur);
        if (strategy == parallelMul)
            CUDA_CREATE_RECORD_EVENT(B_alt_can_be_overwritten, *stream_B_alt);

        if (strategy != simple && strategy != parallelMul)
        {
            copy_matrix_compute_checksum(A, dA_cur, 0, num_split_other_dim, 0, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_cur, 'A', without_error_check);
            CUDA_CREATE_RECORD_EVENT(A_copied, *stream_A_cur);

            copy_matrix_compute_checksum(B, dB_cur, 0, num_split_common_dim, 0, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_cur, 'B', without_error_check);
            CUDA_CREATE_RECORD_EVENT(B_copied, *stream_B_cur);
        }

        for (int C_row = 0; C_row < num_split_other_dim && result_correct; C_row++)
        {
            for (int C_col = 0; C_col < num_split_other_dim && result_correct; C_col += (strategy == parallelMul ? 2 : 1))
            {
                cudaMemsetAsync(dC_cur, 0, size_C_ec, *stream_C_cur);
                if (strategy == parallelMul && C_col + 1 < num_split_other_dim)
                    cudaMemsetAsync(dC_alt, 0, size_C_ec, *stream_C_alt);

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
                        if (strategy == simple)
                        {
                            CUDA_WAIT_DESTROY_EVENT(A_can_be_overwritten, *stream_A_cur, true)
                            copy_matrix_compute_checksum(A, dA_cur, C_row, num_split_other_dim, block, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_cur, 'A', without_error_check);
                            CUDA_CREATE_RECORD_EVENT(A_copied, *stream_A_cur);

                            CUDA_WAIT_DESTROY_EVENT(B_can_be_overwritten, *stream_B_cur, true)
                            copy_matrix_compute_checksum(B, dB_cur, block, num_split_common_dim, C_col, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_cur, 'B', without_error_check);
                            CUDA_CREATE_RECORD_EVENT(B_copied, *stream_B_cur);
                        }
                        else if (strategy == parallelMul)
                        {
                            CUDA_WAIT_DESTROY_EVENT(A_can_be_overwritten, *stream_A_cur, true)
                            copy_matrix_compute_checksum(A, dA_cur, C_row, num_split_other_dim, block, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_cur, 'A', without_error_check);
                            CUDA_CREATE_RECORD_EVENT(A_copied, *stream_A_cur);

                            CUDA_WAIT_DESTROY_EVENT(B_can_be_overwritten, *stream_B_cur, true)
                            copy_matrix_compute_checksum(B, dB_cur, block, num_split_common_dim, C_col, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_cur, 'B', without_error_check);
                            CUDA_CREATE_RECORD_EVENT(B_copied, *stream_B_cur);

                            if (C_col + 1 < num_split_other_dim)
                            {
                                CUDA_WAIT_DESTROY_EVENT(B_alt_can_be_overwritten, *stream_B_alt, true)
                                copy_matrix_compute_checksum(B, dB_alt, block, num_split_common_dim, C_col + 1, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_alt, 'B', without_error_check);
                                CUDA_CREATE_RECORD_EVENT(B_alt_copied, *stream_B_alt);
                            }
                        }
                        else
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

                                CUDA_WAIT_DESTROY_EVENT(A_can_be_overwritten, *stream_A_alt, true)
                                copy_matrix_compute_checksum(A, dA_alt, next_C_row, num_split_other_dim, next_block, num_split_common_dim, rows_A, cols_A, max_block_rows_A, max_block_cols_A, *stream_A_alt, 'A', without_error_check);
                                CUDA_CREATE_RECORD_EVENT(A_alt_copied, *stream_A_alt);

                                CUDA_WAIT_DESTROY_EVENT(B_can_be_overwritten, *stream_B_alt, true)
                                copy_matrix_compute_checksum(B, dB_alt, next_block, num_split_common_dim, next_C_col, num_split_other_dim, ROWS_B, cols_B, MAX_BLOCK_ROWS_B, max_block_cols_B, *stream_B_alt, 'B', without_error_check);
                                CUDA_CREATE_RECORD_EVENT(B_alt_copied, *stream_B_alt);
                            }
                        }

                        CUDA_CHECK
                    }

                    int error_id = block + C_col * num_split_common_dim + C_row * num_split_common_dim * num_split_other_dim;

                    CUDA_WAIT_DESTROY_EVENT(A_copied, *stream_C_cur, strategy != parallelMul || C_col + 1 >= num_split_other_dim)
                    CUDA_WAIT_DESTROY_EVENT(B_copied, *stream_C_cur, true)

                    C_mult_check_correct(dA_cur, dB_cur, dC_cur, rows_A, cols_B, &block_rows_C_cur, &block_cols_C_cur, C_row, C_col, block, max_block_rows_A, max_block_cols_A, max_block_cols_B, *stream_C_cur, *stream_Cbis_cur, num_split_common_dim, num_split_other_dim, errors_count, error_xs[error_id], error_ys[error_id], error_values[error_id], &result_correct, &result_corrected, without_error_check);

                    if (strategy != parallelMul || C_col + 1 >= num_split_other_dim)
                        CUDA_CREATE_RECORD_EVENT(A_can_be_overwritten, *stream_C_cur);
                    CUDA_CREATE_RECORD_EVENT(B_can_be_overwritten, *stream_C_cur);

                    if (strategy == parallelMul && C_col + 1 < num_split_other_dim)
                    {
                        CUDA_WAIT_DESTROY_EVENT(A_copied, *stream_C_alt, true)
                        CUDA_WAIT_DESTROY_EVENT(B_alt_copied, *stream_C_alt, true)

                        error_id = block + (C_col + 1) * num_split_common_dim + C_row * num_split_common_dim * num_split_other_dim;
                        C_mult_check_correct(dA_cur, dB_alt, dC_alt, rows_A, cols_B, &block_rows_C_alt, &block_cols_C_alt, C_row, C_col + 1, block, max_block_rows_A, max_block_cols_A, max_block_cols_B, *stream_C_alt, *stream_Cbis_alt, num_split_common_dim, num_split_other_dim, errors_count, error_xs[error_id], error_ys[error_id], error_values[error_id], &result_correct_alt, &result_corrected_alt, without_error_check);

                        CUDA_CREATE_RECORD_EVENT(A_can_be_overwritten, *stream_C_cur);
                        CUDA_CREATE_RECORD_EVENT(B_alt_can_be_overwritten, *stream_C_alt);
                    }

                    switch (strategy)
                    {
                        case preloadAB_deferUnloadC:
                        case preloadAB:
                            SWAP(dA_cur, dA_alt)
                            SWAP(dB_cur, dB_alt)
                            SWAP(stream_A_cur, stream_A_alt)
                            SWAP(stream_B_cur, stream_B_alt)
                            SWAP(A_copied, A_alt_copied)
                            SWAP(B_copied, B_alt_copied)
                    }
                }
                // send back result (without checksums)

                {
                    ScopedTimer timer("C to host", POST);

                    offset = C_row * MAX_BLOCK_ROWS_C * COLS_C + C_col * MAX_BLOCK_COLS_C;
                    int offset2 = C_row * MAX_BLOCK_ROWS_C * COLS_C + (C_col + 1) * MAX_BLOCK_COLS_C;

                    switch (strategy)
                    {
                        case preloadAB_deferUnloadC:
                            SWAP(dC_cur, dC_alt)
                            SWAP(stream_C_cur, stream_C_alt)
                            SWAP(stream_Cbis_cur, stream_Cbis_alt)
                            SWAP(block_rows_C_cur, block_rows_C_alt)
                            SWAP(block_cols_C_cur, block_cols_C_alt)

                            cp_matrix_from_CUDA(dC_alt, C, block_rows_C_alt, block_cols_C_alt, MAX_BLOCK_COLS_C + extra, offset, COLS_C, *stream_C_alt);
                            break;

                        case parallelMul:
                            if (C_col + 1 < num_split_other_dim)
                                cp_matrix_from_CUDA(dC_alt, C, block_rows_C_alt, block_cols_C_alt, MAX_BLOCK_COLS_C + extra, offset2, COLS_C, *stream_C_alt);

                        case preloadAB:
                        case simple:
                            cp_matrix_from_CUDA(dC_cur, C, block_rows_C_cur, block_cols_C_cur, MAX_BLOCK_COLS_C + extra, offset, COLS_C, *stream_C_cur);
                    }

                    CUDA_CHECK
                }
            }
        }

        // cleanup:

        switch (strategy)
        {
            case preloadAB_deferUnloadC:
            case parallelMul:
                cudaStreamDestroy(stream_C2);
                cudaStreamDestroy(stream_C2bis);

            case preloadAB:
                cudaStreamDestroy(stream_A2);
                cudaStreamDestroy(stream_B2);

            case simple:
                cudaStreamDestroy(stream_A1);
                cudaStreamDestroy(stream_B1);
                cudaStreamDestroy(stream_C1);
                cudaStreamDestroy(stream_C1bis);
        }

        switch (strategy)
        {
            case preloadAB_deferUnloadC:
            case parallelMul:
                cudaFree(dC2);

            case preloadAB:
                if (strategy != parallelMul)
                    cudaFree(dA2);
                cudaFree(dB2);

            case simple:
                cudaFree(dA1);
                cudaFree(dB1);
                cudaFree(dC1);
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
